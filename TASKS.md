# Hand Tracker — Tasks & Status

> Last updated: 2026-04-29

---

## Current State

All core features are implemented and tested. 65 tests pass. The system is ready to use.

The gallery is **empty** — no people have been enrolled yet. Face recognition will show "Unknown" for everyone until you run enrollment.

---

## Completed

- [x] 3-thread pipeline (capture → inference → render)
- [x] Hand skeleton tracking (MediaPipe, 21 landmarks, 2 hands)
- [x] 8 visual filters + pinch-to-cycle gesture
- [x] Middle-finger blur overlay
- [x] Body pose detection + 10 gesture classifiers
- [x] ASL rule-based recognizer (A–Z)
- [x] ASL RandomForest model training script (`models/train_asl.py`)
- [x] DualDetector: RetinaFace + YOLOv8 consensus
- [x] Quality gate (size, score, yaw, blur)
- [x] Face attributes: glasses detection, makeup, quality score
- [x] FaceGallery: .npy storage, cosine search, metadata.json
- [x] EnsembleRecognizer: 3-model embeddings (A/B/C)
- [x] Arbitrator: 5 decision rules, dynamic weights
- [x] TemporalSmoother: 15-frame IoU tracker
- [x] Interactive webcam enrollment (`--mode enroll`)
- [x] Batch enrollment from folder (`--mode enroll --from-folder`)
- [x] Audit mode (`--mode audit`)
- [x] CPU fallback (2-model ensemble without GPU)
- [x] `setup_models.py` model downloader
- [x] 85 unit + integration tests
- [x] README.md, CODEBASE_OVERVIEW.md, SESSION_LOG.md, TASKS.md
- [x] FACE_RECOGNITION_TUTORIAL.md

---

## Required Before First Use

### 1. Install dependencies
```
setup.bat
```
Or manually:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download face recognition models
```
python setup_models.py
```
- InsightFace (buffalo_l + antelopev2): auto-download ~300MB
- YOLOv8n-face: auto-download ~27MB
- AdaFace IR101 (`models/adaface_ir101.onnx`): manual download (see `setup_models.py` output)
- ElasticFace-Arc+ (`models/elasticface_arc.onnx`): optional, manual download

### 3. Enroll people
See `FACE_RECOGNITION_TUTORIAL.md` for the complete walkthrough.

### 4. (Optional) Train ASL model
```
python models/train_asl.py
```
Generates `models/asl_classifier.pkl`. Without it the rule-based fallback still works.

---

## Pending / Nice-to-Have

These are ideas for future improvement — none are blocking current functionality.

### Medium priority

- [ ] **Second-photo re-enrollment**: allow adding a second photo to an existing gallery entry to improve robustness against glasses on/off or different lighting
- [ ] **Enrollment quality feedback on screen**: currently quality warnings only appear in the terminal; show them in the OpenCV window too
- [ ] **Gallery management CLI**: commands to list, rename, or remove enrolled people without manually editing `gallery/metadata.json`
- [ ] **Batch audit after bulk enroll**: currently `run_audit` must be triggered manually; could auto-run after `--from-folder` batch enrollment

### Lower priority

- [ ] **ASL real-data training**: replace synthetic training data in `models/train_asl.py` with a real dataset (e.g., ASL Fingerspelling from Kaggle) for better accuracy on real hands
- [ ] **Multi-camera support**: currently only one camera at a time; could support `--camera 0 --camera 1`
- [ ] **Export recognized names to CSV**: for event-tracking use cases
- [ ] **Dynamic threshold tuning**: GUI slider to adjust cosine similarity thresholds live
- [ ] **DeepSORT tracker**: replace simple IoU tracker with DeepSORT for better identity persistence across occlusions at crowded events

### Known limitations

- On CPU without GPU, effective FPS is ~3–5 fps with 5 faces in frame (2-model ensemble)
- ASL rule-based recognizer has ~70% accuracy on real hands; the ML model improves this but needs real training data
- `mediapipe` is pinned to `<=0.10.14` — upgrading will break the legacy `solutions` API used for hands/pose
- ElasticFace-Arc+ requires manual ONNX download; system degrades gracefully to 2-model without it
