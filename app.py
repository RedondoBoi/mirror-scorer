import io
import os
import urllib.request
from typing import List

import numpy as np
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import mediapipe as mp
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Face detector model file is downloaded once and cached on disk. The beauty
# classifier is loaded via the transformers library, which downloads its own
# weights + matched config the first time it's used and caches them too.
# ---------------------------------------------------------------------------
CACHE_DIR = "model_cache"

FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_DETECTOR_PATH = os.path.join(CACHE_DIR, "blaze_face_short_range.tflite")

BEAUTY_MODEL_REPO = "Aruno/gemini-beauty"

# Maps this model's class labels to a 1-5 beauty value. Read dynamically
# from the model's own config.id2label at load time rather than assuming a
# fixed index order.
LABEL_VALUES = {
    "very_ugly": 1.0,
    "very ugly": 1.0,
    "ugly": 2.0,
    "normal": 3.0,
    "attractive": 4.0,
    "very_attractive": 5.0,
    "very attractive": 5.0,
}

_face_detector = None
_beauty_model = None
_beauty_processor = None


def _ensure_cached(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        _ensure_cached(FACE_DETECTOR_URL, FACE_DETECTOR_PATH)
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=FACE_DETECTOR_PATH),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        _face_detector = FaceDetector.create_from_options(options)
    return _face_detector


def get_beauty_model():
    global _beauty_model, _beauty_processor
    if _beauty_model is None:
        _beauty_processor = AutoImageProcessor.from_pretrained(BEAUTY_MODEL_REPO)
        _beauty_model = MobileNetV2ForImageClassification.from_pretrained(BEAUTY_MODEL_REPO)
        _beauty_model.eval()
    return _beauty_model, _beauty_processor


def _crop_face(pil_img: Image.Image, bbox) -> Image.Image:
    w, h = pil_img.size
    pad_x = int(bbox.width * 0.25)
    pad_y = int(bbox.height * 0.25)

    left = max(0, bbox.origin_x - pad_x)
    top = max(0, bbox.origin_y - pad_y)
    right = min(w, bbox.origin_x + bbox.width + pad_x)
    bottom = min(h, bbox.origin_y + bbox.height + pad_y)

    return pil_img.crop((left, top, right, bottom)).convert("RGB")


def _score_face_crop(face_img: Image.Image) -> float:
    """Returns a 1-5 beauty value from the classifier's class probabilities."""
    model, processor = get_beauty_model()

    inputs = processor(images=face_img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=0).tolist()

    id2label = model.config.id2label

    total = 0.0
    weight_sum = 0.0
    for idx, prob in enumerate(probs):
        raw_label = id2label.get(idx) or id2label.get(str(idx)) or ""
        key = raw_label.strip().lower()
        value = LABEL_VALUES.get(key) or LABEL_VALUES.get(key.replace(" ", "_"))
        if value is None:
            continue
        total += prob * value
        weight_sum += prob

    if weight_sum <= 0:
        raise RuntimeError(f"No recognized labels in model.config.id2label: {id2label}")

    return total / weight_sum


def score_one_image(image_bytes: bytes):
    pil_img = Image.open(io.BytesIO(image_bytes))
    # Phone photos carry an EXIF rotation tag rather than storing pixels
    # already rotated — without this, a sideways/upside-down image gets fed
    # straight to the face detector and no face is found.
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    np_img = np.ascontiguousarray(np.array(pil_img))

    diag = {"width": pil_img.width, "height": pil_img.height}

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)

    detector = get_face_detector()
    result = detector.detect(mp_image)

    diag["num_detections"] = len(result.detections) if result.detections else 0

    if not result.detections:
        return None, diag

    # use the largest detected face, in case more than one face is in frame
    best = max(
        result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
    )

    face_img = _crop_face(pil_img, best.bounding_box)

    raw_1_5 = _score_face_crop(face_img)
    raw_1_5 = max(1.0, min(5.0, raw_1_5))
    score_0_100 = (raw_1_5 - 1.0) / 4.0 * 100.0
    return score_0_100, diag


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug_models")
def debug_models():
    info = {}
    try:
        get_face_detector()
        info["face_detector_loaded"] = True
    except Exception as e:
        info["face_detector_loaded"] = False
        info["face_detector_error"] = f"{type(e).__name__}: {e}"

    try:
        model, _ = get_beauty_model()
        info["beauty_model_loaded"] = True
        info["beauty_model_labels"] = model.config.id2label
    except Exception as e:
        info["beauty_model_loaded"] = False
        info["beauty_model_error"] = f"{type(e).__name__}: {e}"

    if os.path.exists(FACE_DETECTOR_PATH):
        info["face_detector_file"] = {"exists": True, "bytes": os.path.getsize(FACE_DETECTOR_PATH)}
    else:
        info["face_detector_file"] = {"exists": False}

    return info


@app.post("/score_upload")
async def score_upload(photos: List[UploadFile] = File(...)):
    if not photos:
        raise HTTPException(status_code=400, detail={"error": "no_photos"})

    scores: List[float] = []
    skipped = 0
    per_photo_debug = []

    for i, photo in enumerate(photos):
        content = await photo.read()
        if not content or len(content) < 500:
            skipped += 1
            per_photo_debug.append({"photo": i, "error": "file_too_small", "bytes": len(content or b"")})
            continue
        try:
            s, diag = score_one_image(content)
        except Exception as e:
            s, diag = None, {"exception": f"{type(e).__name__}: {e}"}

        if s is None:
            skipped += 1
            per_photo_debug.append({"photo": i, "error": "no_score", **diag})
        else:
            scores.append(s)
            per_photo_debug.append({"photo": i, "score": s, **diag})

    if not scores:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no_face_detected",
                "photos_received": len(photos),
                "debug": per_photo_debug,
            },
        )

    final_score = sum(scores) / len(scores)

    return {
        "ok": True,
        "score_0_100": round(final_score, 2),
        "photos_scored": len(scores),
        "photos_skipped": skipped,
        "debug": per_photo_debug,
    }