
import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Configuration
DATA_ROOT = Path('Data/')        # root directory containing person subfolders
OUTPUT_ROOT = Path('dataset/')   # where train/val/test folders will be created
TARGET_PERSON = 'Vaibhav'         # name of the target folder under DATA_ROOT

# Split ratios (must sum to 1)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1"

# Define transforms
train_aug_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor()
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Gather file paths
all_person_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
target_dir = DATA_ROOT / TARGET_PERSON
other_dirs = [d for d in all_person_dirs if d.name != TARGET_PERSON]

# Collect all target images
target_images = list(target_dir.rglob('*.jpg'))
n_target = len(target_images)
if n_target == 0:
    raise ValueError(f"No .jpg images found under {target_dir}")

# Collect all other images
all_other_images = []
for person_dir in tqdm(other_dirs, desc="Gathering other images"):
    all_other_images.extend(person_dir.rglob('*.jpg'))

if len(all_other_images) < n_target:
    raise ValueError("Not enough images from other people to match the target count.")

# Randomly sample the same number of "other" images
random.seed(42)
other_images = random.sample(all_other_images, n_target)

# Create labeled list
data = [(p, TARGET_PERSON) for p in target_images] + [(p, 'Other') for p in other_images]
labels = [label for _, label in data]

# First split: train vs temp (val+test)
train_data, temp_data, _, temp_labels = train_test_split(
    data,
    labels,
    test_size=VAL_RATIO + TEST_RATIO,
    stratify=labels,
    random_state=42
)

# Second split: val vs test
val_fraction = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
val_data, test_data, _, _ = train_test_split(
    temp_data,
    temp_labels,
    test_size=1 - val_fraction,
    stratify=temp_labels,
    random_state=42
)

# Helper to apply transforms and copy files
def process_and_save_image(src_path, dest_path, is_train=False):
    img = Image.open(src_path).convert('RGB')  # RGB only (no grayscale)

    if is_train:
        img = train_aug_transforms(img)
    else:
        img = val_test_transforms(img)

    # Save tensor back as image
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(dest_path)

# Helper to process datasets
def copy_subset(subset_name, subset_data, is_train=False):
    for src_path, label in tqdm(subset_data, desc=f"Copying {subset_name}"):
        rel_path = src_path.relative_to(
            DATA_ROOT / (TARGET_PERSON if label == TARGET_PERSON else src_path.parents[len(src_path.parents)-1].name)
        )
        dest_dir = OUTPUT_ROOT / subset_name / label / rel_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        process_and_save_image(src_path, dest_path, is_train=is_train)

# Perform copying
copy_subset('train', train_data, is_train=True)
copy_subset('val', val_data, is_train=False)
copy_subset('test', test_data, is_train=False)

print("Dataset split and augmentation complete! ✅")
