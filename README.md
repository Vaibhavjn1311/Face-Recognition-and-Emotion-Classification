# SMAI Assignment 2: Face and Emotion Recognition

## Overview
This assignment covers essential statistical methods in artificial intelligence, focusing on face recognition and emotion classification tasks. You will use binary classification for face recognition and multiclass classification for emotion recognition. 

## Setup & Requirements
- Use WandB for metrics and plots
- Handle potential CUDA memory constraints by adjusting batch size, freezing layers, or optimizing memory management

## Part 1: Face Recognition (Binary Classification)
### Task
Classify images into two categories:
- **Your Face (Label 1)**
- **Not Your Face (Label 0)**

### Data Collection
- Collect balanced datasets under varying conditions (light, background, occlusions)
- Prepare separate training and challenging test sets
- Implement data augmentation (flips, rotations, jitter)

### Model Architectures
1. **VGGFace (Fine-tuning)**
   - Pretrained VGGFace with modified final layer for binary classification
   - Run:
   ```
   python ./VGG_Q1.py
   ```
   - On running this it will make a checkpoint file which is : vgg16_face_classifier.pth
   - After than to run the model:
   ```
   python ./inference.py --model-path vgg16_face_classifier.pth
   ```
2. **ResNet18 (Scratch)**
   - Fully trained ResNet18 from random initialization
   - Run:
   ```
   python ./resnet18_scratch_Q1.py
   ```
   - On running this it will make a checkpoint file which is : resnet18_scratch_face.pth
   - After than to run the model:
   ```
   python ./inference_resnet18.py --model-path resnet18_scratch_face.pth
   ```
3. **ResNet18 (Pretrained on ImageNet)**
   - ImageNet pretrained ResNet18 with modified final layer
    - Run:
   ```
   python ./resnet18_pretrained_Q1.py
   ```
   - On running this it will make a checkpoint file which is : resnet18_pretrained_face.pth
   - After than to run the model:
   ```
   python ./inference_resnet18.py --model-path resnet18_pretrained_face.pth
   ```

## Part 2: Emotion Recognition (Multiclass Classification)
### Task
Classify emotions in face images into at least three categories (happy, sad, neutral, etc.).

### Data
- Use images from Part 1 with appropriate emotion labels
- Apply the same data augmentations

### Model Architectures
Adapt architectures from Part 1 with final layers tailored for k-class emotion recognition.
1. **VGGFace (Fine-tuning)**
   - Run:
   ```
   python ./vgg_emotion.py
   ```
   - On running this it will make a checkpoint file which is : best_vggface.pth
   - After than to run the model, first you have to change the model: "vgg" and checkpoint: "best_vggface.pth" in live_camera_inference.py and then :
   ```
   python ./live_camera_inference.py
   ```
2. **ResNet18 (Scratch)**
   - Run:
   ```
   python ./resnet_scratch_emotion.py
   ```
   - On running this it will make a checkpoint file which is : best_resnet_scratch.pth
   - After than to run the model, first you have to change the model: "resnet_scratch" and checkpoint: "best_resnet_scratch.pth" in live_camera_inference.py and then :
   ```
   python ./live_camera_inference.py
   ```
3. **ResNet18 (Pretrained on ImageNet)**
    - Run:
   ```
   python ./resnet_pretrained_emotion.py
   ```
   - On running this it will make a checkpoint file which is : resnet18_pretrained_face.pth
   - After than to run the model, first you have to change the model: "resnet_pretrained" and checkpoint: "best_resnet_pretrained.pth" in live_camera_inference.py and then :
   ```
   python ./live_camera_inference.py
   ```



