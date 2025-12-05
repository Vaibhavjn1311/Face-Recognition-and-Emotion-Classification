
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16_bn
from sklearn.metrics import classification_report
from tqdm import tqdm
import wandb

# ---------------------------------------------------------------------
class BinaryLabel:
    def __init__(self, pos_idx):
        self.pos_idx = pos_idx

    def __call__(self, y):
        return 1 if y == self.pos_idx else 0

# ---------------------------------------------------------------------
def build_vgg16_imagenet(num_classes=2, freeze_features=True):
    model = vgg16_bn(pretrained=True)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, num_classes)
    return model

# ---------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    loop = tqdm(loader, desc="  Train", leave=False)
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss  = F.cross_entropy(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)

        loop.set_postfix(loss=f"{running_loss/len(loader.dataset):.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total

    wandb.log({
        "Train Loss": epoch_loss,
        "Train Accuracy": epoch_acc,
        "epoch": epoch
    })

    return epoch_loss

# ---------------------------------------------------------------------
def eval_model(model, loader, device, phase="Val", epoch=None):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    loop = tqdm(loader, desc=f"  {phase}", leave=False)
    with torch.no_grad():
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            lbls = preds.argmax(1)
            correct += (lbls == yb).sum().item()
            total   += yb.size(0)
            all_preds  += lbls.cpu().tolist()
            all_labels += yb.cpu().tolist()
            loop.set_postfix(acc=f"{correct/total:.4f}")

    acc = correct / total

    if phase != "Test" and epoch is not None:
        wandb.log({
            f"{phase} Accuracy": acc,
            "epoch": epoch
        })

    return acc, all_preds, all_labels

# ---------------------------------------------------------------------
def main():
    # Initialize wandb
    wandb.init(project="face-recognition-vgg16", config={
        "model": "VGG16_bn",
        "epochs": 3,
        "batch_size": 32,
        "optimizer": "Adam",
        "lr": 1e-4,
        "image_size": 224,
        "loss_fn": "cross_entropy",
        "task": "Binary Face Classification"
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Paths
    base_dir   = "dataset"
    train_dir  = os.path.join(base_dir, "train")
    val_dir    = os.path.join(base_dir, "val")
    test_dir   = os.path.join(base_dir, "test")

    # Label mapping
    raw = ImageFolder(train_dir)
    mapping = raw.class_to_idx
    target_tf = BinaryLabel(mapping['Vaibhav'])

    # Datasets & Loaders
    train_ds = ImageFolder(train_dir, transform=train_tf, target_transform=target_tf)
    val_ds   = ImageFolder(val_dir,   transform=val_tf,   target_transform=target_tf)
    test_ds  = ImageFolder(test_dir,  transform=val_tf,   target_transform=target_tf)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Model
    model = build_vgg16_imagenet(num_classes=2, freeze_features=True)
    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )

    # Training loop
    for epoch in tqdm(range(1, config.epochs+1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_acc, _, _ = eval_model(model, val_loader, device, phase="Val", epoch=epoch)
        print(f"Epoch {epoch}/{config.epochs} — Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Test evaluation
    test_acc, preds, labels = eval_model(model, test_loader, device, phase="Test")
    print(f"\n=== Test Accuracy: {test_acc:.4f} ===\n")
    print(classification_report(labels, preds, target_names=["Other", "Vaibhav"]))

    wandb.log({"Test Accuracy": test_acc})

    # Save model
    save_path = "vgg16_face_classifier.pth"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)
    print(f"✅ Model weights saved to {save_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
