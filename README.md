# BFINet

Official Pytorch Code base for "Irregular agricultural field delineation using a dual-branch architecture from high-resolution remote sensing images"

[Project](https://github.com/NanNanmei/BFINet)

## Introduction

We propose a boundary-field interaction network, namely BFINet, leveraging multitask learning techniques for AF delineation. BFINet comprises two branches: a core branch for AF delineation, and an auxiliary branch for boundary prediction that furnishes fine-grained boundary information to enhance geometric feature learning.

## Using the code:

The code is stable while using Python 3.9.0, CUDA >=11.0

- Clone this repository:
```bash
git clone https://github.com/NanNanmei/BFINet
cd BFINet
```

To install all the dependencies using conda or pip:

```
PyTorch
TensorboardX
OpenCV
numpy
tqdm
```

## Preprocessing
Using the code preprocess.py and image_crop.py to obtain boundary maps and sample patches, respectively. 

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <train>
    ├── image
    |   ├── 001.tif
    │   ├── 002.tif
    │   ├── 003.tif
    │   ├── ...
    |
    └── mask
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── contour
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── ...
```

For testing and validation datasets, the same structure as the above.

## Training and testing

Our code will release after our paper is accepted.

## A pretrained weight
A pretrained weight of PVT-V2 on the ImageNet dataset is provided: https://drive.google.com/file/d/1uzeVfA4gEQ772vzLntnkqvWePSw84F6y/view?usp=sharing

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [BsiNet](https://github.com/long123524/BsiNet-torch), [SEANet](https://github.com/long123524/SEANet_torch), and [HGINet](https://github.com/long123524/HGINet-torch)

### Citation:
If you find this work useful or interesting, please consider citing the following references.
```
[1] Zhao H, Long J, Zhang M, et.al. Irregular agricultural field delineation using a dual-branch architecture from high-resolution remote sensing images. IEEE GEOSCIENCE AND REMOTE SENSING LETTERS.
[2] Long J, Li M, Wang X, et.al. Delineation of agricultural fields using multi-task BsiNet from high-resolution satellite images. International Journal of Applied Earth Observation and Geoinformation, 2022, 112:102871.
[3] Li M, Long J, Stein A, et.al. sing a semantic edge-aware multi-task neural network to delineate agricultural parcels from remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing, 2023, 200:24-40.

```
