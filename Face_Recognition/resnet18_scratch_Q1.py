
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from sklearn.metrics import classification_report
from tqdm import tqdm
import wandb  # ← wandb added

# -----------------------------------------------------------------------------
# 1) Binary label mapping
# -----------------------------------------------------------------------------
class BinaryLabel:
    def __init__(self, pos_idx):
        self.pos_idx = pos_idx
    def __call__(self, y):
        return 1 if y == self.pos_idx else 0

# -----------------------------------------------------------------------------
# 2) ResNet18 from scratch
# -----------------------------------------------------------------------------
def build_resnet18_scratch(num_classes=2):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -----------------------------------------------------------------------------
# 3) Training & evaluation
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="  Train", leave=False)
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        loop.set_postfix(loss=f"{total_loss/len(loader.dataset):.4f}")
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device, phase="Val"):
    model.eval()
    correct, total = 0, 0
    preds_all, labels_all = [], []
    loop = tqdm(loader, desc=f"  {phase}", leave=False)
    with torch.no_grad():
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            preds_all += preds.cpu().tolist()
            labels_all += yb.cpu().tolist()
            loop.set_postfix(acc=f"{correct/total:.4f}")
    return correct / total, preds_all, labels_all

# -----------------------------------------------------------------------------
# 4) Main
# -----------------------------------------------------------------------------
def main():
    wandb.init(project="resnet_face_classifier", name="resnet18_scratch")
    config = wandb.config
    config.epochs = 10
    config.lr = 1e-4
    config.batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Paths
    base_dir = "dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    # Dataset + Labeling
    raw = ImageFolder(train_dir)
    mapping = raw.class_to_idx
    target_tf = BinaryLabel(mapping['Vaibhav'])

    train_ds = ImageFolder(train_dir, transform=train_tf, target_transform=target_tf)
    val_ds   = ImageFolder(val_dir,   transform=val_tf,   target_transform=target_tf)
    test_ds  = ImageFolder(test_dir,  transform=val_tf,   target_transform=target_tf)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Model & optimizer
    model = build_resnet18_scratch(num_classes=2).to(device)
    wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    for epoch in tqdm(range(1, config.epochs+1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = eval_model(model, val_loader, device, phase="Val")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch}/{config.epochs} — Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Final test eval
    test_acc, preds, labels = eval_model(model, test_loader, device, phase="Test")
    wandb.log({"test_acc": test_acc})
    print(f"\n=== Test Accuracy: {test_acc:.4f} ===\n")
    print(classification_report(labels, preds, target_names=["Other", "Vaibhav"]))

    # Save model
    save_path = "resnet18_scratch_face.pth"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)
    print("Model saved to:", save_path)

if __name__ == "__main__":
    main()
