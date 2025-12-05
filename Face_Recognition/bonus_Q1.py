# bonus_Q1_resnet.py

import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ─── USER PARAMETERS ──────────────────────────────────────────────────────────
MODEL_PATH       = "resnet18_pretrained_face.pth"
TEST_DIR         = "Bonus_q1_dataset"     # your dataset root
OUTPUT_VIDEO     = "output.mp4"
FPS              = 1                  # frames per second
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────────────────────────────────────────────

def find_images(topdir):
    """
    Recursively walk `topdir` and collect every image file beneath it.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = []
    if not os.path.isdir(topdir):
        print(f"⚠️  Warning: '{topdir}' not found. No images will be loaded.")
        return imgs

    for root, _, files in os.walk(topdir):
        for fn in files:
            if fn.lower().endswith(exts):
                imgs.append(os.path.join(root, fn))
    imgs = sorted(imgs)
    print(f"🔍 Found {len(imgs)} images under '{topdir}'.")
    return imgs


def build_resnet18(num_classes=2):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(path, device):
    model = build_resnet18(num_classes=2)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def make_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])


def predict_image(model, img_path, device, transform):
    img = Image.open(img_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        return int(logits.argmax(dim=1).item())


def create_video(image_paths, labels, output_path, fps):
    """Write out a video with 'Locked'/'Unlocked' overlaid on each frame."""
    # detect frame size
    for p in image_paths:
        frame = cv2.imread(p)
        if frame is not None:
            h, w = frame.shape[:2]
            break
    else:
        print("❌ No valid images for video.")
        return

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    if not writer.isOpened():
        print("❌ Couldn't open VideoWriter.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (path, lbl) in enumerate(zip(image_paths, labels), 1):
        frame = cv2.imread(path)
        if frame is None:
            continue
        status = "Unlocked" if lbl == 1 else "Locked"
        color  = (0,200,0) if lbl == 1 else (0,0,200)
        cv2.putText(frame, status, (10,30), font, 1.0, color, 2, cv2.LINE_AA)
        writer.write(frame)
        if idx % 50 == 0 or idx == len(image_paths):
            print(f"  🎞  Wrote {idx}/{len(image_paths)} frames…")

    writer.release()
    print(f"✅ Video saved to '{output_path}'.\n")


def main():
    # 1) collect all images
    images = find_images(TEST_DIR)
    if not images:
        return

    # 2) load model + transform
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Checkpoint not found: {MODEL_PATH}")
        return
    model     = load_model(MODEL_PATH, DEVICE)
    transform = make_transform()

    # 3) run inference on every image
    labels = []
    for path in images:
        lbl = predict_image(model, path, DEVICE, transform)
        labels.append(lbl)

    # 4) print per‐image status
    for path, lbl in zip(images, labels):
        rel = os.path.relpath(path)
        print(f"{rel}: {'Unlocked' if lbl==1 else 'Locked'}")
    print(f"\n✅ Processed {len(images)} images.\n")

    # 5) stitch into a video
    create_video(images, labels, OUTPUT_VIDEO, FPS)


if __name__ == "__main__":
    main()
