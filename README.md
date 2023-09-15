#  Solution of Team luojc for FLARE23 Challenge

## Introduction

### Overview of our work.

![image](https://github.com/Prech-start/FLARE23_AdaptNet/blob/main/IMG/overview.png)

## Environments and Installation

The basic language for our work is [python](https://www.python.org/), and the baseline
is [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). So, you can install the nnunet frame with
the [GitHub Repository](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), or use the following comments:

```
pip install torch torchvision torchaudio
pip install -e .
```

## Usage

### convert CT images to npy

we modify the normalization function with ___preprocessing.py___,
and you could use the following comments to processing the CT images:

```
python nnunet/experiment_planning/nnUNet_convert_decathlon_task.py -i [FLARE23_imageTr_path]

python nnunet/experiment_planning/nnUNet_plan_and_preprocess -t [TASK_ID] --verify_dataset_integrity
```

It must be noted that the method is based on the __nnU-Net__, so I recommend you to convert the dataset within nnU-Net's
data preprocessing.

The usage and note concerning for ___nnUNet_convert_decathlon_task.py___ is recorded
on [website](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md).

After preprocessing, we will obtain several folders:

```
- nnU-Net_base_folder
    - nnUNet_prepocessing
    - nnUNetFrame
    - nnUNet_raw
    - nnUNet_trained_models
```

### generate and process pseudo labels and update dataset

We use the [method](https://github.com/Ziyan-Huang/FLARE22) to generate the pseudo labels.

Then, to Simplify the processes, you can use the following code to convert the dataset:

```
python data_convert.py -pseudo_label_folder -nnunet_preprocessing_folder -imagesTr_floder -labelTr_floder
```
where the __nnunet_preprocessing_folder__ is the folder path of the dataset planed by nnunet. like 'nnU-Net_base_folder/nnUNet_preprocessed/Task098_FLARE2023/nnUNetData_plans_v2.1_stage1'

## Training

```
python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 DATA_ID FOLD[0,1,2,3,4] --npz --disable_saving --disable_validation_inference
```

## Inference
```
python inference.py [INPUT_FOLDER] [OUTPUT_FOLDER]
```
Before the Inference, you should move the best nnunet checkpoints to replace the three files in folder __'checkpoints'__.

## Reference

MACCAI FLARE2023 https://codalab.lisn.upsaclay.fr/competitions/12239

MACCAI FLARE2022 Team balackbean https://github.com/Ziyan-Huang/FLARE22

## Citations

## What's New?

we have fix the [problem](https://github.com/Prech-start/FLARE23_AdaptNet/blob/a81cbd4463fccce56fff8cdca3828aade2a4f66d/utils/utils.py#L318) the influence from data dtype which may effects the final result!


