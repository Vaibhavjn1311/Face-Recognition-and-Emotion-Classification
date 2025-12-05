import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import wandb

# -------------------------------------
# Configuration
# -------------------------------------
DATA_ROOT = 'dataset'      # Root directory of train/val folders
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS = ['Happy', 'Sad', 'Neutral', 'Angry']
CHECKPOINT_PATH = 'best_resnet_pretrained.pth'

# -------------------------------------
# Dataset Definition
# -------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        for idx, emot in enumerate(EMOTIONS):
            base = os.path.join(root_dir, split, 'Vaibhav', emot)
            for cond in os.listdir(base):
                cond_dir = os.path.join(base, cond)
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
def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

# -------------------------------------
# Model Factory
# -------------------------------------
def get_resnet_pretrained():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
    return model.to(DEVICE)

# -------------------------------------
# Training & Validation with Checkpointing
# -------------------------------------
def train_and_validate():
    # Initialize W&B
    wandb.init(
        project='emotion_resnet_pretrained',
        config={
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'model': 'ResNet18-Pretrained'
        }
    )

    # Data loaders
    train_tf, val_tf = get_transforms()
    train_loader = DataLoader(
        EmotionDataset(DATA_ROOT, 'train', train_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        EmotionDataset(DATA_ROOT, 'val', val_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = get_resnet_pretrained()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
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
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        # Log metrics
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_acc': val_acc})
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved improved checkpoint: {CHECKPOINT_PATH} (Val Acc: {best_val_acc:.4f})")

    wandb.finish()

# -------------------------------------
# Entry Point
# -------------------------------------
if __name__ == '__main__':
    train_and_validate()
