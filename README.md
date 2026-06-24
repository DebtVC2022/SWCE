# SWCE

Code for the paper:

**Segmented Weighting Cross-Entropy Loss for Robust Learning with Noisy Labels**

## Code Overview

The code files are listed in the order in which the corresponding experiments
are presented in the paper.

| File | Role in the paper |
| --- | --- |
| `swce_common.py` | Shared implementation of SWCE, Focal Loss, noise generation, transforms, model construction, training, evaluation, and CSV utilities |
| `benchmark_cifar_synthetic_noise.py` | Standard CIFAR-10/100 synthetic-noise benchmark for CE vs SWCE |
| `benchmark_tinyimagenet_cifarn.py` | Tiny-ImageNet synthetic-noise and CIFAR-N human-noise benchmark |
| `benchmark_cifar_extended_baselines.py` | CIFAR baseline comparison for CE, Focal Loss, SWCE, Co-teaching, and MentorNet-PD |
| `benchmark_official_dividemix.py` | Wrapper for the official `LiJunnan1992/DivideMix` CIFAR code; pass the official code directory through `--official-root`; the wrapper uses an isolated runtime copy |
| `diagnostic_cifar_ablation_beta.py` | CIFAR diagnostic experiments for beta stability, component ablation, and noise-identification checks |
| `summarize_results.py` | Summarizes benchmark statistics, diagnostic analyses, and the CIFAR 60% high-noise check |
| `application_image_folder.py` | ImageFolder application-dataset experiment for Cat-dog, Chest X-ray, MRI brain, and Traffic Sign data |

## Data

External image datasets are not included in this repository. They can be
obtained from the public sources below. The datasets are listed in the order in
which they appear in the paper.

| Dataset | Source |
| --- | --- |
| CIFAR-10 and CIFAR-100 | Official CIFAR dataset: https://www.cs.toronto.edu/~kriz/cifar.html |
| Tiny-ImageNet-200 | Stanford CS231n Tiny-ImageNet release: http://cs231n.stanford.edu/tiny-imagenet-200.zip |
| CIFAR-N human noisy labels | Official CIFAR-N release: https://github.com/UCSC-REAL/cifar-10-100n |
| Cat and Dog Image Dataset | Kaggle: https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset |
| Chest X-ray Image Dataset | Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| MRI brain image dataset | Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| Traffic Sign Image Dataset | Kaggle: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection |

## Configuration

| Item | Requirement |
| --- | --- |
| Python | 3.9 or 3.10 |
| Core libraries | PyTorch, torchvision, NumPy, pandas |
| Optional analysis dependency | SciPy |
| Optional baseline dependency | torchnet, only if running the official DivideMix baseline |
