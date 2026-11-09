# ResNet-18 CIFAR-10 Classification — Experimental Results

## Overview
Three variants of the same reduced-width ResNet-18 were trained from scratch for 300 epochs on a CIFAR-10 image-classification dataset, changing only the training-image augmentation strategy to compare their effects on accuracy.

All three models use the same modified ResNet-18 architecture with **4,368,690 parameters**, cross-entropy loss, **SGD** (learning rate 0.1, momentum 0.9, weight decay 0.0005), and a **cosine annealing** learning-rate schedule. The stage widths **(40, 80, 160, 320)** refer to the **number of output feature channels** in the network’s four successive residual stages (`layer1`, `layer2`, `layer3`, and `layer4`), respectively—not the number of layers. Each stage contains **two residual blocks**: `layer1` outputs 40-channel feature maps, `layer2` outputs 80-channel maps, `layer3` outputs 160-channel maps, and `layer4` outputs 320-channel maps. These reduced widths replace the standard ResNet-18 widths **(64, 128, 256, 512)** to keep the model below the project’s 5-million-parameter limit.

## Experiment Setups

- **Baseline:** Randomly crop 32 × 32 images with four pixels of padding and apply random horizontal flips.
- **RandAugment:** Apply the baseline augmentations plus `torchvision.transforms.RandAugment(num_ops=2, magnitude=5)`.
- **ColorJitter + RandomErasing:** Apply the baseline augmentations plus random brightness, contrast, and saturation adjustments (`0.2` each), and randomly erase small rectangular regions (`p=0.25`, `scale=(0.02, 0.10)`).

## Results

| Training augmentation | Maximum training accuracy | Maximum validation accuracy | Kaggle public test accuracy |
|---|---:|---:|---:|
| Baseline | 99.90% | 95.32% | 84.5% |
| **RandAugment** | 99.72% | **95.56%** | **84.9%** |
| ColorJitter + RandomErasing | 99.90% | 95.32% | 83.9% |

**Best result:** RandAugment achieved the highest maximum validation accuracy (**95.56%**) and Kaggle public test accuracy (**84.9%**) of the three experiments.

*Note: Kaggle public test accuracy reflects the public leaderboard evaluation, not necessarily the final private leaderboard score. Training and validation figures are the maximum accuracies observed during each run.*

## GitHub Repositories
- **Reference implementation:** [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) — the PyTorch CIFAR ResNet implementation used as a starting reference for the modified architecture.
