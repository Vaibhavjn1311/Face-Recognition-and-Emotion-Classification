
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
import wandb  # ← added wandb

# -----------------------------------------------------------------------------
# 1) Pickle‑safe binary label mapper
# -----------------------------------------------------------------------------
class BinaryLabel:
    def __init__(self, pos_idx):
        self.pos_idx = pos_idx
    def __call__(self, y):
        return 1 if y == self.pos_idx else 0

# -----------------------------------------------------------------------------
# 2) Model builder: ResNet‑18 pretrained on ImageNet
# -----------------------------------------------------------------------------
def build_resnet18_pretrained(num_classes=2, freeze_backbone=False):
    model = resnet18(pretrained=True)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

# -----------------------------------------------------------------------------
# 3) Training / Evaluation Helpers
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="  Train", leave=False)
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        loop.set_postfix(loss=f"{running_loss/len(loader.dataset):.4f}")
    return running_loss / len(loader.dataset)

def eval_model(model, loader, device, phase="Val"):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    loop = tqdm(loader, desc=f"  {phase}", leave=False)
    with torch.no_grad():
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds  = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
            all_preds  += preds.cpu().tolist()
            all_labels += yb.cpu().tolist()
            loop.set_postfix(acc=f"{correct/total:.4f}")
    return correct/total, all_preds, all_labels

# -----------------------------------------------------------------------------
# 4) Main Training Script
# -----------------------------------------------------------------------------
def main():
    # ─── WandB Setup ───
    wandb.init(project="resnet_face_classifier", name="resnet18_pretrained")
    config = wandb.config
    config.epochs = 3
    config.lr = 1e-4
    config.weight_decay = 1e-4
    config.batch_size = 32
    config.freeze_backbone = False

    # ─── Device ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─── Transforms ───
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    # ─── Paths ───
    base_dir  = "dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    # ─── Label mapping ───
    raw = ImageFolder(train_dir)
    mapping = raw.class_to_idx
    target_tf = BinaryLabel(mapping['Vaibhav'])

    # ─── Datasets & Loaders ───
    train_ds = ImageFolder(train_dir, transform=train_tf, target_transform=target_tf)
    val_ds   = ImageFolder(val_dir,   transform=val_tf, target_transform=target_tf)
    test_ds  = ImageFolder(test_dir,  transform=val_tf, target_transform=target_tf)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=4)

    # ─── Model + Optimizer ───
    model = build_resnet18_pretrained(num_classes=2, freeze_backbone=config.freeze_backbone)
    model.to(device)
    wandb.watch(model, log="all")  # optional: log gradients/weights

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config.lr, weight_decay=config.weight_decay)

    # ─── Training Loop ───
    for epoch in tqdm(range(1, config.epochs+1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = eval_model(model, val_loader, device, phase="Val")

        # ─── Log to wandb ───
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': val_acc
        })
        print(f"Epoch {epoch}/{config.epochs} — Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # ─── Final Evaluation ───
    test_acc, preds, labels = eval_model(model, test_loader, device, phase="Test")
    wandb.log({"test_acc": test_acc})
    print(f"\n=== Test Accuracy: {test_acc:.4f} ===\n")
    print(classification_report(labels, preds, target_names=["Other","Vaibhav"]))

    # ─── Save model ───
    save_path = "resnet18_pretrained_face.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    wandb.save(save_path)

if __name__ == "__main__":
    main()
