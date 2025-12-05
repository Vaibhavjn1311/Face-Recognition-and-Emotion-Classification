import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image
from tqdm import tqdm
import wandb

# -------------------------------------
# Configuration
# -------------------------------------
DATA_ROOT      = 'dataset'           # root folder containing train/val/test
BATCH_SIZE     = 32
NUM_EPOCHS     = 50                  # bumped up so scheduler and ES can kick in
LEARNING_RATE  = 1e-4
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS       = ['Happy', 'Sad', 'Neutral', 'Angry']
NUM_CLASSES    = len(EMOTIONS)
CHECKPOINT_PATH = 'best_vggface.pth'
ES_PATIENCE    = 7                   # early‐stop after 7 epochs without improvement
LR_FACTOR      = 0.5                 # factor to reduce LR on plateau
LR_PATIENCE    = 3                   # scheduler patience

# -------------------------------------
# Dataset Definition
# -------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.images  = []
        self.labels  = []
        for idx, emot in enumerate(EMOTIONS):
            emot_dir = os.path.join(root_dir, split, 'Vaibhav', emot)
            if not os.path.isdir(emot_dir):
                raise FileNotFoundError(f"Directory not found: {emot_dir}")
            for cond in os.listdir(emot_dir):
                cond_dir = os.path.join(emot_dir, cond)
                if not os.path.isdir(cond_dir):
                    continue
                for fname in os.listdir(cond_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(cond_dir, fname))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# -------------------------------------
# Transforms
# -------------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------------
# Model Factory
# -------------------------------------
def create_vgg_model():
    # load ImageNet‐pretrained VGG16 then swap to VGGFace if weights exist
    model = vgg16(pretrained=True)
    weights_path = 'vggface_weights.pth'
    if os.path.isfile(weights_path):
        print(f"Loading VGG‐Face weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path))
    else:
        print("VGG‐Face weights not found, using ImageNet pretrained backbone")
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model.to(DEVICE)

# -------------------------------------
# Training & Validation
# -------------------------------------
def train_and_validate():
    # Initialize W&B
    wandb.init(
        project='emotion_vggface',
        config={
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'model': 'VGGFace',
            'es_patience': ES_PATIENCE,
            'lr_factor': LR_FACTOR,
            'lr_patience': LR_PATIENCE
        }
    )

    # Data loaders
    train_loader = DataLoader(
        EmotionDataset(DATA_ROOT, 'train', train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        EmotionDataset(DATA_ROOT, 'val', val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model, loss & optimizer
    model     = create_vgg_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: reduce LR when val_acc plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=True
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        # ---- Scheduler & Early Stopping Logic ----
        scheduler.step(val_acc)  # adjust LR
        current_lr = optimizer.param_groups[0]['lr']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f" → New best val_acc: {best_val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f" ↳ No improvement for {epochs_no_improve}/{ES_PATIENCE} epochs.")

        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr
        })

        # Early stop
        if epochs_no_improve >= ES_PATIENCE:
            print(f"Early stopping triggered after {ES_PATIENCE} epochs without improvement.")
            break

        print(f"Epoch {epoch}/{NUM_EPOCHS} • "
              f"Loss: {avg_train_loss:.4f} • Val Acc: {val_acc:.4f} • LR: {current_lr:.1e}")

    wandb.finish()
    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")

# -------------------------------------
# Main Entry Point
# -------------------------------------
if __name__ == '__main__':
    train_and_validate()
