import io
import os
import urllib.request
from typing import List

import numpy as np
import torch

# A single CPU thread uses meaningfully less memory than torch's default
# (which tries to use all available cores) — worth it on a memory-constrained
# instance even though it costs a little speed.
torch.set_num_threads(1)
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import mediapipe as mp
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
from facenet_pytorch import InceptionResnetV1

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
_face_embedder = None

# Starting threshold for "same person" on L2 distance between embeddings —
# this is a commonly-cited starting point for this exact model, not
# something we've tuned on our own data. Expect to revisit once real users
# have gone through this — lighting, angle, and glasses all shift the
# distance even for genuine matches.
VERIFICATION_DISTANCE_THRESHOLD = 1.0


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


def get_face_embedder():
    global _face_embedder
    if _face_embedder is None:
        # Half precision roughly halves this model's memory footprint
        # (107MB -> ~53MB) — worth it given how memory-constrained this
        # instance is. Basic conv/batchnorm/linear ops used here are
        # well-supported in fp16 on CPU.
        _face_embedder = InceptionResnetV1(pretrained="vggface2").eval().half()
    return _face_embedder


def _face_embedding(face_img: Image.Image) -> torch.Tensor:
    """Turns a cropped face into a 512-number embedding, using this model's
    documented preprocessing convention (160x160, fixed_image_standardization)."""
    resized = face_img.resize((160, 160))
    arr = np.asarray(resized).astype(np.float32)
    # This specific formula (not ImageNet mean/std) is what this model's
    # pretrained weights expect — it's the library's own "prewhitening" step.
    arr = (arr - 127.5) / 128.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).half()

    model = get_face_embedder()
    with torch.no_grad():
        embedding = model(tensor)
    return embedding[0].float()


def _l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item()


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


def _load_and_crop_face(image_bytes: bytes):
    """Loads an image, corrects EXIF rotation, detects the largest face, and
    returns (face_crop, diag) — face_crop is None if no face was found."""
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
    return face_img, diag


def score_one_image(image_bytes: bytes):
    face_img, diag = _load_and_crop_face(image_bytes)
    if face_img is None:
        return None, diag

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

    try:
        get_face_embedder()
        info["face_embedder_loaded"] = True
    except Exception as e:
        info["face_embedder_loaded"] = False
        info["face_embedder_error"] = f"{type(e).__name__}: {e}"

    if os.path.exists(FACE_DETECTOR_PATH):
        info["face_detector_file"] = {"exists": True, "bytes": os.path.getsize(FACE_DETECTOR_PATH)}
    else:
        info["face_detector_file"] = {"exists": False}

    return info


@app.post("/verify_selfie")
async def verify_selfie(selfie: UploadFile = File(...), photos: List[UploadFile] = File(...)):
    """Checks whether the live selfie matches the face in each scoring
    photo. This is meant to run BEFORE scoring — a real identity gate, not
    an informational check."""
    selfie_bytes = await selfie.read()
    if not selfie_bytes or len(selfie_bytes) < 500:
        raise HTTPException(status_code=400, detail={"error": "selfie_too_small"})

    selfie_face, selfie_diag = _load_and_crop_face(selfie_bytes)
    if selfie_face is None:
        raise HTTPException(status_code=400, detail={"error": "no_face_in_selfie", "debug": selfie_diag})

    selfie_embedding = _face_embedding(selfie_face)

    per_photo = []
    any_match = False

    for i, photo in enumerate(photos):
        content = await photo.read()
        if not content or len(content) < 500:
            per_photo.append({"photo": i, "error": "file_too_small"})
            continue

        face, diag = _load_and_crop_face(content)
        if face is None:
            per_photo.append({"photo": i, "error": "no_face_detected", **diag})
            continue

        embedding = _face_embedding(face)
        distance = _l2_distance(selfie_embedding, embedding)
        match = distance < VERIFICATION_DISTANCE_THRESHOLD

        if match:
            any_match = True

        per_photo.append({"photo": i, "distance": round(distance, 4), "match": match})

    return {
        "ok": True,
        "verified": any_match,
        "threshold": VERIFICATION_DISTANCE_THRESHOLD,
        "per_photo": per_photo,
    }


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
