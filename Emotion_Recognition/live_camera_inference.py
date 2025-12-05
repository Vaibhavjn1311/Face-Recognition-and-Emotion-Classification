import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS = ['Happy', 'Sad', 'Neutral', 'Angry']
MODEL_TYPE = 'resnet_scratch'  # 'vgg', 'resnet_pretrained', 'resnet_scratch'
CHECKPOINT = 'best_resnet_scratch.pth'

# Map each emotion to a custom message
message_map = {
    'Happy':   "Keep smiling!!",
    'Sad':     "It's okay to feel blue.",
    'Neutral': "Steady as she goes.",
    'Angry':   "Take a deep breath..."
}

# Load model
if MODEL_TYPE == 'vgg':
    from torchvision.models import vgg16
    model = vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, len(EMOTIONS))
elif MODEL_TYPE == 'resnet_pretrained':
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
elif MODEL_TYPE == 'resnet_scratch':
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
else:
    raise ValueError('Unknown MODEL_TYPE')

model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Run webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL and preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    inp = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        out = model(inp)
        pred = out.argmax(dim=1).item()
    label = EMOTIONS[pred]
    message = message_map.get(label, "")

    # Overlay emotion label
    cv2.putText(frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2)

    # Overlay the descriptive message below the label
    cv2.putText(frame,
                message,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,0),
                2)

    cv2.imshow('Live Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
