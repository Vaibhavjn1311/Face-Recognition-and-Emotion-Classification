import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16_bn, VGG16_BN_Weights

# ─────────────────────────────────────────────────────────────
# 1) Model builder
# ─────────────────────────────────────────────────────────────
def build_vgg16_imagenet(num_classes: int = 2, freeze_features: bool = True) -> nn.Module:
    """
    Returns a VGG‑16‑BN backbone with the final FC layer swapped to `num_classes`.
    If `freeze_features` is True, conv weights are frozen.
    """
    # start from *random* init; use `weights=VGG16_BN_Weights.IMAGENET1K_V1`
    # if you actually want ImageNet pre‑training
    model = vgg16_bn(weights=None)

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False

    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, num_classes)
    return model


# ─────────────────────────────────────────────────────────────
# 2) Letterbox resize helper
# ─────────────────────────────────────────────────────────────
def letterbox_image(img: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize & pad PIL image to `size` while keeping aspect ratio."""
    iw, ih = img.size
    w,  h  = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    img_resized = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new("RGB", size, (0, 0, 0))
    new_img.paste(img_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_img


# ─────────────────────────────────────────────────────────────
# 3) Pre‑processing pipeline
# ─────────────────────────────────────────────────────────────
norm_tf = transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])

pre_tf = transforms.Compose([
    transforms.ToTensor(),
    norm_tf
])

def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    OpenCV BGR numpy array  →  1×3×224×224 float tensor, normalized.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = letterbox_image(img_pil, size=(224, 224))
    return pre_tf(img_pil).unsqueeze(0)  # add batch dim


# ─────────────────────────────────────────────────────────────
# 4) Robust model loader
# ─────────────────────────────────────────────────────────────
def load_model(model_path: str,
               device: torch.device,
               num_classes: int = 2,
               freeze_features: bool = True) -> nn.Module:
    """
    Handles all common checkpoint formats:
    • torch.save(model, ...)
    • torch.save(model.state_dict(), ...)
    • torch.save({'model': model.state_dict(), ...}, ...)
    """
    # build bare architecture
    model = build_vgg16_imagenet(num_classes=num_classes,
                                 freeze_features=freeze_features)

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        # if dict wrapper, unwrap
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────
# 5) Inference loop
# ─────────────────────────────────────────────────────────────
def main(model_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(model_path, device)

    class_names = ["Other", "Vaibhav"]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW fixes some Windows cams
    if not cap.isOpened():
        print("❌ Unable to access webcam")
        return

    print("🔎  Webcam running — press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            inp = preprocess_frame(frame).to(device)

            with torch.no_grad():
                logits = model(inp)
                probs  = F.softmax(logits, dim=1)[0]
                idx    = torch.argmax(probs).item()
                conf   = probs[idx].item() * 100

            label = f"{class_names[idx]}: {conf:.1f}%"
            cv2.putText(frame, label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Webcam Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# 6) CLI entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to the VGG‑16 checkpoint (.pth/.pt)")
    args = parser.parse_args()
    main(args.model_path)
