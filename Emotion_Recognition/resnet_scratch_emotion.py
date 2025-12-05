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
data_root      = 'dataset'
batch_size     = 32
num_epochs     = 50            # give scheduler & ES time to work
lr             = 1e-4
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotions       = ['Happy', 'Sad', 'Neutral', 'Angry']
num_classes    = len(emotions)
CHECKPOINT_PATH= 'best_resnet_scratch.pth'

# Early stopping & LR schedule
ES_PATIENCE  = 7               # stop after this many epochs w/o val_acc gain
LR_FACTOR    = 0.5             # multiply LR by this on plateau
LR_PATIENCE  = 3               # how many epochs to wait before dropping LR

# Initialize W&B
wandb.init(
    project='emotion_resnet_scratch',
    config={
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'model': 'ResNet18-Scratch',
        'es_patience': ES_PATIENCE,
        'lr_factor': LR_FACTOR,
        'lr_patience': LR_PATIENCE
    }
)

# -------------------------------------
# Dataset Definition
# -------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        for idx, emot in enumerate(emotions):
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

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]

# -------------------------------------
# Transforms
# -------------------------------------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------------------
# Main Training Loop
# -------------------------------------
if __name__ == '__main__':
    # DataLoaders
    train_loader = DataLoader(
        EmotionDataset(data_root, 'train', train_tf),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        EmotionDataset(data_root, 'val', val_tf),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model: ResNet18 from scratch (or pretrained if you uncomment)
    model = models.resnet18(pretrained=False)
    # model = models.resnet18(pretrained=True)  # ← try this for finetuning!
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # LR scheduler: drop LR on plateau of val_acc
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=True
    )

    best_val_acc     = 0.0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Val]   Epoch {epoch}"):
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        # ---- Scheduler & Early Stopping ----
        scheduler.step(val_acc)  
        current_lr = optimizer.param_groups[0]['lr']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f" → New best val_acc: {best_val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f" ↳ No improvement for {epochs_no_improve}/{ES_PATIENCE} epochs.")

        # ---- Log to W&B ----
        wandb.log({
            'epoch':        epoch,
            'train_loss':   train_loss,
            'val_acc':      val_acc,
            'learning_rate': current_lr
        })

        print(f"Epoch {epoch}/{num_epochs} • "
              f"Loss: {train_loss:.4f} • Val Acc: {val_acc:.4f} • LR: {current_lr:.1e}")

        if epochs_no_improve >= ES_PATIENCE:
            print(f"Early stopping triggered after {ES_PATIENCE} epochs without improvement.")
            break

    wandb.finish()
    print(f"Finished. Best val_acc = {best_val_acc:.4f}")
