# configurable-leakage-controlled-multi-workflow-medical-image-classification-suite

## Main settings:

Start with **Section 0, cell 3: Settings, package installation and shared helper functions**. Go through the settings below in order, change the values you need, and then choose **Run all** in Google Colab. `True` means enabled; `False` means disabled. A segmentation, or mask, marks the tumour area in an image.

**The original segmentation is always included in the selected classification workflows. CAS2 and nnINTERACTIVE are optional, additional resegmentations. They do not replace the original segmentation or its analysis.** You can compare results from the original masks with results from one or both additional mask types.

In this code, **CAS2 uses SAM 2.1**. Its switches therefore contain `SAM2`, while its output folder is named `CAS2`.

### 1. Development data — `DEVELOPMENT_ROOT`

```python
DEVELOPMENT_ROOT = [
    "/content/drive/MyDrive//DATA/DEVELOPMENT-150",
    "/content/drive/MyDrive/DATA/none",
]
```

This setting gives the folder or list of folders containing the patients used to develop the models. Several folders are combined into **one development dataset**; they are read one after another. Splitting a large dataset between folders may reduce the very high memory use seen when first reading approximately 350 or more files from one folder.

### 2. Independent test data — `TEST_ROOT`

```python
TEST_ROOT = [
    "/content/drive/MyDrive/DATA/TEST-40",
    "/content/drive/MyDrive/none",
]
```

Choose **one or two separate test datasets**. These patients are used to evaluate the final model, not to train it. The final model is trained once using all development patients, then evaluated once on each test dataset. Each test dataset has its own result tables and is named after its folder.

For both development and test data:

- Google Colab uses `/content/drive/MyDrive` for the Google Drive folder called **My Drive**.
- The code searches the selected folders and their subfolders.
- Each patient's NRRD image and matching NRRD mask must be in the same folder.
- Duplicate patients across folders stop the run.
- A folder path containing `none`, in any combination of upper- and lowercase letters, is skipped with a warning. You can use this to disable an optional folder. Each setting must still contain at least one real folder.

### 3. Results folder — `OUT_ROOT`

```python
OUT_ROOT = "/content/drive/MyDrive/DATA/RESULTS/RESULTS-AUTOsegm"
```

This is where the code saves result tables, extracted image features in CSV files, model files and overview tables. Choose a separate results folder for each analysis you want to keep.

### 4. Patient outcomes — `LABELS_CSV`

```python
LABELS_CSV = "/content/drive/MyDrive/DATA/labels.csv"
```

This selects the CSV file containing the outcome you want the model to predict. **Column A contains `patient_id`; column B contains the outcome as `0` or `1`.** These two numbers represent the two outcome groups in your study.

Each run uses one label file. To study another outcome, change `LABELS_CSV` and run the notebook again. Also change `OUT_ROOT` to preserve earlier summary tables.

### 5. Image slices per patient — `N_SLICES`

```python
N_SLICES = 3
```

This selects how many image directions are used for each patient:

| Value | Directions used |
| --- | --- |
| `1` | Axial: a horizontal slice through the body. |
| `2` | Axial and sagittal: horizontal and side-view slices. |
| `3` | Axial, sagittal and coronal: horizontal, side-view and front-view slices. |

For each direction, the code selects the slice with the largest tumour cross-section. The supplied setting is `3`.

### 6. Analysis workflows — three independent switches

```python
RUN_FROZEN_CNN_E2E   = True
RUN_CNN_LR           = False
RUN_CNN_FINETUNE_E2E = True
```

Enable the workflows you want to compare. A CNN is a neural network that learns from images. Its backbone is the main part that extracts image information.

| Setting | What it enables | Supplied value |
| --- | --- | --- |
| `RUN_FROZEN_CNN_E2E` | Section 1, `frozen`: trains a classifier while keeping the pretrained CNN backbone unchanged. | `True` |
| `RUN_CNN_LR` | Section 2, `efficient_ML`: extracts image features using an unchanged pretrained CNN, then uses these features in classical machine-learning models. | `False` |
| `RUN_CNN_FINETUNE_E2E` | Section 3, `finetune`: trains a classifier and also adjusts the pretrained CNN backbone using the development data. | `True` |

Setting a switch to `False` skips that workflow. These choices apply to the original-mask analysis and any enabled additional-mask analyses.

### 7. CNN models — `SHARED_BACKBONES`

```python
SHARED_BACKBONES = [
#   "radimagenet_densenet121",
#   "imagenet_densenet121",
#   "radimagenet_inceptionv3",
#   "imagenet_inceptionv3",
    "radimagenet_resnet10t",
    "imagenet_resnet10t",
#   "radimagenet_resnet18",
#   "imagenet_resnet18",
#   "radimagenet_resnet50",
#   "imagenet_resnet50",
#   "choi_idh",
]
```

This list selects the CNN models used across the three workflows. A line beginning with `#` is inactive. Remove the `#` to include a model; add it to exclude a model.

The supplied list enables `radimagenet_resnet10t` and `imagenet_resnet10t`. These use the same named CNN architecture but different pretrained weights. The list also offers DenseNet121, InceptionV3, ResNet18, ResNet50 and `choi_idh` options.

Sections 1 and 3 report and skip architectures they do not support. Section 2 extracts features with all architectures listed as active. Instructions for downloading RadImageNet weights are provided below.

### 8. Additional segmentations and mask comparison — Section 0, cell 5

```python
RUN_AUTOSEG_NNINTERACTIVE = True
RUN_AUTOSEG_SAM2          = False

ANALYZE_NNINTERACTIVE_MASKS = True
ANALYZE_SAM2_MASKS          = False
```

The code can generate additional tumour masks using **nnINTERACTIVE** and **CAS2 / SAM 2.1**. It compares each generated mask with the original mask using the **Dice Similarity Coefficient (DSC)**. A DSC of `1` means the masks match exactly; `0` means they have no overlap. This measures agreement with the original mask, not proof that either mask is correct.

The existing masks provide the guidance, called **prompts**, for generating the new masks. These are therefore guided resegmentations, not segmentations made independently of the original masks.

- **nnINTERACTIVE** works with the 3D image. It uses one lasso prompt on one slice per tumour. A lasso is an outline drawn around the tumour area.
- **CAS2 / SAM 2.1** processes 2D slices. It uses one rectangular box around the tumour on every slice containing the tumour.

Generation and classification analysis have separate switches:

| Setting | What it enables | Supplied value |
| --- | --- | --- |
| `RUN_AUTOSEG_NNINTERACTIVE` | Generates or loads nnINTERACTIVE masks and compares them with the original masks. | `True` |
| `RUN_AUTOSEG_SAM2` | Generates or loads CAS2 / SAM 2.1 masks and compares them with the original masks. | `False` |
| `ANALYZE_NNINTERACTIVE_MASKS` | Adds the selected classification workflows using nnINTERACTIVE masks. | `True` |
| `ANALYZE_SAM2_MASKS` | Adds the selected classification workflows using CAS2 / SAM 2.1 masks. | `False` |

To generate and analyse an additional mask type, set both of its switches to `True`. To compare its masks without adding classification analyses, enable its `RUN_AUTOSEG` switch and disable its `ANALYZE` switch. Setting both `RUN_AUTOSEG` switches to `False` skips the segmentation-comparison cell and its extra package installation; leave both `ANALYZE` switches `False` when you want only the original-mask analysis.

**With the supplied settings, classification uses both original and nnINTERACTIVE masks. CAS2 is disabled. To include all three mask types, set all four switches to `True`. The original-mask analysis remains included in every case.**

### 9. nnINTERACTIVE guidance — `NNINTERACTIVE_ANALYSIS_PROMPT_TYPE`

```python
NNINTERACTIVE_ANALYSIS_PROMPT_TYPE = "lasso"  # fixed; other prompt types are unsupported
```

Keep this setting as `"lasso"`. This version supports only lasso guidance for nnINTERACTIVE generation and analysis. Other prompt types are not supported.

### 10. Saved segmentation folders — `CAS2_ROOT` and `LASSO_ROOT`

```python
CAS2_ROOT = "/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/CAS2"
LASSO_ROOT = "/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/nninteraction-lasso"
```

Generated masks are saved permanently as one NRRD file per patient in Google Drive:

| Setting | Supplied folder | Mask type |
| --- | --- | --- |
| `CAS2_ROOT` | `/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/CAS2` | CAS2 / SAM 2.1 |
| `LASSO_ROOT` | `/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/nninteraction-lasso` | nnINTERACTIVE with lasso guidance |

Before generating a mask, the code checks whether its saved file already exists. Existing masks are loaded and reused; missing masks are generated and saved. Keeping the same folders allows later runs to reuse the masks and saves processing time.

These folders store the **additional masks**. `OUT_ROOT` stores the **analysis results**. The original image and mask files remain in the development and test folders.

---

A highly configurable Google Colab notebook comparing three leakage-controlled DL and ML workflows for medical image classification. Featuring a single-notebook design with clear top-level settings, it is easy to use for non-coding researchers yet fully customizable for experienced developers.

Repository Contents:
This repository consists of two primary components:

Code_1 (DICOM or MRB to NRRD Conversion): A script that converts raw DICOM or MRB data into paired NRRD image and mask files.

Code_2 (Analysis Pipeline): The main workflow notebook that inputs the created NRRD pairs of image and mask files for analysis.

CNN WEIGHTS
Download the pretrained RadImageNet weights for all CNN architectures and place the files in the directory specified by WEIGHTS_BASE:
WEIGHTS_BASE = "/content/drive/MyDrive/DATA/weights"
You can change WEIGHTS_BASE in script 2 to point to any directory in your Google Drive.

DenseNet121
https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/DenseNet121.pt?download=true

ResNet50
https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt?download=true

ResNet18
https://huggingface.co/convergedmachine/RadImagenet/resolve/main/resnet18.pth?download=true

ResNet10t
https://huggingface.co/convergedmachine/RadImagenet/resolve/main/resnet10t.pth?download=true

InceptionV3
https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/InceptionV3.pt?download=true



Further Instructions and Settings
Detailed instructions for settings and code use are provided in the comments within the code itself.

How to Cite
If you use this code in a published study, please cite it to ensure proper academic credit. You can use the "Cite this repository" button on the right side of the GitHub page or use the following reference:

(submitted for publication)
[name/authors]. (2026). [Repository Name]. GitHub. [GitHub URL]
