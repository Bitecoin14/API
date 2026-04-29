# Face Recognition — Enrollment Tutorial

How to feed photos into the system so it can recognise people.

---

## Overview

There are two ways to enroll people:

| Method | Best for | Command |
|--------|----------|---------|
| **Folder of photos** | 100+ people, pre-taken photos | `--mode enroll --from-folder ./photos/` |
| **Webcam capture** | One person at a time, live | `--mode enroll` |

### Which recognition system will be used?

| InsightFace installed? | What happens |
|------------------------|--------------|
| **Yes** | High-accuracy 3-model ensemble. Embeddings stored in `gallery/`. |
| **No** | Automatic fallback to OpenCV LBPH. Photos copied into `known_faces/<name>/`. |

Either way, enrollment just works. Install InsightFace later and re-enroll to upgrade accuracy.

---

## Before You Start

### 1. Install dependencies and models

```bat
setup.bat
python setup_models.py
```

`setup_models.py` downloads InsightFace and YOLOv8 automatically (~330 MB total). AdaFace and ElasticFace require manual ONNX download — the script prints instructions if those files are missing. The system works fine with only InsightFace (Model A) if the others aren't available.

### 2. Verify your camera works

```
python hand_tracker.py
```

You should see the hand-tracking window. Press Q to close.

---

---

## Method 1: Webcam Enrollment (One Person at a Time)

Best when the person is physically present.

```
python hand_tracker.py --mode enroll
```

### Step-by-step

**Step 1 — Enter the name**

The terminal will prompt:
```
Enter person's name (or 'q' to quit):
```
Type the full name as you want it displayed, e.g. `Alice Johnson`. Press Enter.

**Step 2 — Frame the face**

A camera window opens. Position the person's face so it:
- Fills roughly 1/3 of the frame width (not too far back)
- Faces the camera directly (less than 35° sideways turn)
- Is well-lit with no harsh shadows or backlighting
- Is not blurry (keep still)

A green rectangle appears when a face is detected.

**Step 3 — Capture**

Press **SPACE** when the face looks good.

The system checks quality automatically. If it rejects the capture, it will say why:
- `"No face detected"` — move closer or improve lighting
- `"Multiple faces detected"` — ensure only one person is in frame
- `"Quality too low (0.28)"` — improve lighting, reduce blur, face camera directly

**Step 4 — Confirm**

A preview window shows the 112×112 aligned crop that will be stored. Close it (any key).

The terminal prompts:
```
Enrolled 'Alice Johnson'. Is this correct? (y/n/r):
```
- **y** — save to gallery and continue to the next person
- **n** — discard and re-enter the name
- **r** — retake the photo for the same name (keeps the name, re-enters capture loop)

**Step 5 — Confusable-pair warning**

After saving, the system compares the new person's embedding against everyone already enrolled. If two people look similar, you'll see:
```
⚠ WARNING: 'Alice Johnson' looks similar to 'Carol Smith'
  Scores → model_a: 0.41  model_b: 0.38
  Consider enrolling a second photo for both.
```
This is informational — the enrollment still succeeds. It means the system may occasionally confuse these two people under poor lighting.

**Step 6 — Enroll the next person**

The terminal prompts for another name. Type `q` when done.

---

## Method 2: Batch Enrollment from a Folder

Best for enrolling many people before an event using pre-taken photos.

### Prepare the photos

1. Take or collect one clear, well-lit, frontal-facing photo per person.
2. Name each photo after the person (the filename becomes the display name):

```
rui.jpg           →  "Rui"
alice_johnson.jpg →  "Alice Johnson"
bob-smith.png     →  "Bob Smith"
carol white.jpg   →  "Carol White"
```

Underscores, hyphens, and spaces all become spaces. Each word is capitalised.

3. **Drag the photos into the `photos/` folder** inside the project directory.

```
Hand Tracking\
  photos\
    rui.jpg
    alice_johnson.jpg
    bob_smith.png
    ...
```

### Photo requirements

| Requirement | Target | Why |
|-------------|--------|-----|
| Face visible | Required | Obviously |
| Face width in photo | ≥ 80px | Below this the detector can't get reliable landmarks |
| Only one face per photo | Required | Multiple faces will cause the image to be skipped |
| Face angle (yaw) | ≤ 35° sideways | Profile shots degrade embedding quality |
| Blur | Low | Sharp images produce better embeddings |
| Lighting | Even | Avoid harsh side-lighting or backlighting |
| Expression | Neutral or slight smile | Extreme expressions can affect embeddings |

**Ideal photo:** passport-style, looking straight at camera, well-lit, neutral expression.

**Acceptable:** casual portrait, slight angle, light smile.

**Avoid:** side profiles, sunglasses, heavy hats/shadows, small faces in group shots.

### Run enrollment

Once the photos are in `photos/`, just run:

```
python hand_tracker.py --mode enroll
```

No `--from-folder` needed — the program detects the photos automatically.

For each image the system will:
1. Detect the face (uses lenient thresholds — suitable for real portraits)
2. Show a 224×224 preview crop for 500ms
3. Prompt in the terminal:

```
Enroll 'Alice Johnson' from alice_johnson.jpg? (y/n/r/a):
```

- **y** — enroll and continue
- **n** — skip this image
- **r** — enter a custom name (useful if the filename is wrong)
- **a** — auto-enroll all remaining images without prompting (good for 100+ photos)

Images that fail quality checks are skipped automatically with a reason:
```
✗ No face detected: blurry_photo.jpg
! 2 faces in group_shot.jpg — using highest quality
✗ Quality too low (0.05): very_dark_photo.jpg
```

---

## After Enrollment: Run the Audit

Once all people are enrolled, run:

```
python hand_tracker.py --mode audit
```

This computes similarity scores between every pair of enrolled people and flags anyone who looks similar enough to cause confusion:

```
═══════════════════════════════════
      CONFUSABLE PAIRS REPORT
═══════════════════════════════════

WARNING: Alice Johnson ↔ Carol Smith
   model_a: 0.41  model_b: 0.38
   Recommendation: consider enrolling a second photo for both.

HIGH RISK: David Lee ↔ Daniel Lee
   model_a: 0.52  model_b: 0.49
   Recommendation: require unanimous consensus for these identities.

═══════════════════════════════════
```

**HIGH RISK** (similarity > 0.45): The system will require all three models to agree before displaying either name. It's stricter but safer.

**WARNING** (similarity > 0.35): The system will use the standard majority-voting rules but you're warned.

The audit writes these flags into `gallery/metadata.json` automatically — no further action needed. Just run the audit and the stricter rules apply immediately in live mode.

---

## Live Face Recognition

After enrollment and audit, run:

```
python hand_tracker.py --mode face
```

Or combined with hand tracking:

```
python hand_tracker.py --face
```

### What you'll see

Each detected face gets a colored bounding box:

| Color | Status | Meaning |
|-------|--------|---------|
| **Green** | CONFIRMED | All models agree, all scores strong |
| **Cyan-yellow** | SOFT_CONFIRMED | 2 out of 3 models agree, strong scores |
| **Orange** | LOW_CONFIDENCE | Models agree but scores are weak — treat as tentative |
| **Red** | AMBIGUOUS | Models disagree — person might be between two enrolled people |
| **Grey** | UNKNOWN | Person not in gallery, or too far/blurry/angled |

Names appear after 3 consecutive consistent recognitions (prevents flicker on first detection).

---

## Gallery Management

### View enrolled people

The gallery is stored in `gallery/metadata.json`. Open it in any text editor to see who is enrolled and when.

### Remove a person

Delete their `.npy` files from `gallery/embeddings/` and remove their entry from `gallery/metadata.json`.

Naming convention: `firstname_lastname_model_a.npy`, `firstname_lastname_model_b.npy`, etc.

Example — to remove "Alice Johnson":
- Delete `gallery/embeddings/alice_johnson_model_a.npy`
- Delete `gallery/embeddings/alice_johnson_model_b.npy`
- Delete `gallery/embeddings/alice_johnson_model_c.npy` (if it exists)
- Remove the `"alice_johnson"` key from `gallery/metadata.json`

Then re-run `--mode audit` to update confusable-pair flags.

### Re-enroll a person

Simply run enrollment again with the same name. The new embeddings will overwrite the old `.npy` files and metadata.

---

## Tips for Best Results

1. **One clear photo per person is enough.** The system creates 5 augmented variants automatically (original, flip, brightness ±15%, slight blur). You don't need to provide multiple photos.

2. **Lighting matters most.** Harsh shadows across the face, backlighting, or colored light (e.g. prom strobes) reduce accuracy. Enroll under similar lighting to the recognition environment when possible.

3. **Always run audit after batch enrollment.** It takes 10 seconds and can prevent embarrassing misidentifications.

4. **For twins or very similar-looking people:** the system will flag them as HIGH RISK. The safest option is to tell the operator which one is which and rely on context rather than automated recognition.

5. **Glasses on at enrollment, off at event (or vice versa):** the attribute layer detects this change and boosts ElasticFace (Model C) weight. The system handles it, but accuracy drops — consider enrolling a second photo with/without glasses if it's a known concern.

6. **For large batches (100+ people):** use `--from-folder` for speed. Sort your photos into the folder first, then run enrollment all at once. The audit at the end is fast regardless of gallery size.
