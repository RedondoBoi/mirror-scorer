import io
import os
import urllib.request
from typing import List

import numpy as np
import torch
import torchvision.models as tv_models
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import mediapipe as mp

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
# Model files are downloaded once on first request and cached on disk for
# the lifetime of the Render instance. Nothing needs to be bundled by hand.
# ---------------------------------------------------------------------------
CACHE_DIR = "model_cache"

FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_DETECTOR_PATH = os.path.join(CACHE_DIR, "blaze_face_short_range.tflite")

BEAUTY_MODEL_URL = (
    "https://huggingface.co/Gustrd/SCUT-FBP5500-PyTorch-Model/"
    "resolve/main/resnet18_py3.pth"
)
BEAUTY_MODEL_PATH = os.path.join(CACHE_DIR, "resnet18_py3.pth")

_beauty_model = None
_face_detector = None


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
    global _beauty_model
    if _beauty_model is None:
        _ensure_cached(BEAUTY_MODEL_URL, BEAUTY_MODEL_PATH)
        m = tv_models.resnet18(num_classes=1)
        state_dict = torch.load(BEAUTY_MODEL_PATH, map_location="cpu")
        m.load_state_dict(state_dict)
        m.eval()
        _beauty_model = m
    return _beauty_model


FACE_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop_face(pil_img: Image.Image, bbox) -> Image.Image:
    w, h = pil_img.size
    pad_x = int(bbox.width * 0.25)
    pad_y = int(bbox.height * 0.25)

    left = max(0, bbox.origin_x - pad_x)
    top = max(0, bbox.origin_y - pad_y)
    right = min(w, bbox.origin_x + bbox.width + pad_x)
    bottom = min(h, bbox.origin_y + bbox.height + pad_y)

    return pil_img.crop((left, top, right, bottom)).convert("RGB")


def _face_to_tensor(face_img: Image.Image) -> torch.Tensor:
    face_img = face_img.resize((FACE_INPUT_SIZE, FACE_INPUT_SIZE))
    arr = np.asarray(face_img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr).unsqueeze(0)


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
    tensor = _face_to_tensor(face_img)

    model = get_beauty_model()
    with torch.no_grad():
        raw = model(tensor).item()

    # SCUT-FBP5500 beauty ratings are on a ~1-5 scale
    raw = max(1.0, min(5.0, raw))
    score_0_100 = (raw - 1.0) / 4.0 * 100.0
    return score_0_100, diag


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug_models")
def debug_models():
    # Confirms the model files actually downloaded (and how big they are),
    # without needing to score anything.
    info = {}
    try:
        get_face_detector()
        info["face_detector_loaded"] = True
    except Exception as e:
        info["face_detector_loaded"] = False
        info["face_detector_error"] = f"{type(e).__name__}: {e}"

    try:
        get_beauty_model()
        info["beauty_model_loaded"] = True
    except Exception as e:
        info["beauty_model_loaded"] = False
        info["beauty_model_error"] = f"{type(e).__name__}: {e}"

    for name, path in [
        ("face_detector_file", FACE_DETECTOR_PATH),
        ("beauty_model_file", BEAUTY_MODEL_PATH),
    ]:
        if os.path.exists(path):
            info[name] = {"exists": True, "bytes": os.path.getsize(path)}
        else:
            info[name] = {"exists": False}

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