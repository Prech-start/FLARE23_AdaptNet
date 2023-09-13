# FLARE23 AdaptNet for MICCAI FLARE2023 Challenge

## Introduction

## Environments and Installation

## Usage

### convert CT images to npy

The baseline module is nnU-Net, and it must convert data to npy files follow
code [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

After preprocessing, we will obtain several folders:
```
-- nnU-Net_base_folder
    -- nnUNet_prepocessing
    -- nnUNetFrame
    -- nnUNet_raw
    -- nnUNet_trained_models
```

### generate and process pseudo labels

### update cases info

For Simplifying the processes, you can use the following code to convert the dataset:

```
python data_convert.py -PSEUDO_LABEL_PATH --NNUNET_npy_PATH
```

It must be noted that the method is based on the nnU-Net, so I recommend you to convert the dataset within nnU-Net's
data preprocessing.

## Training

```
python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 DATA_ID FOLD[0,1,2,3,4] --npz --disable_saving --disable_validation_inference
```

## Inference