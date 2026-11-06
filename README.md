# configurable-leakage-controlled-multi-workflow-medical-image-classification-suite

A highly configurable Google Colab notebook that compares three leakage-controlled deep-learning (DL) and machine-learning (ML) workflows for medical image classification. "Leakage-controlled" means the test patients are never used for training or for choosing settings. Everything is in one notebook, and the main settings are grouped at the top. Researchers who do not write code can use it, and experienced developers can still change every part of it.

**What is new in this version**

| New function | What it does for you | Where to read more |
| --- | --- | --- |
| **Two ways to load data** | Load NRRD image/mask pairs from development and test folders, **or** load NIfTI files from separate image and segmentation folders. You choose with one switch. | Settings 1–2 |
| **Contour perturbation test** | Checks how much the results change when the tumour outline is drawn slightly differently (larger, smaller, shifted or rescaled). | Setting 13 |
| **nnINTERACTIVE and SAM 2 resegmentation** | Creates automatic tumour masks, compares them with the manual masks and reports how well they agree (Dice Similarity Coefficient, DSC). This gives a measure of manual-versus-automatic (inter-observer) variability. The automatic masks can also be used for the full classification analysis. | Settings 11–12 |
| **New CNN models** | Adds DINOv3 Vision Transformers (`dinov3_vits16`, `dinov3_vitb16`) and the MRI-pretrained 3D model MARS (`mars_3d`), as well as ImageNet versions of the existing models. | Setting 9 |
| **Other additions** | Patient include/exclude column, lesion-size limit, choice between the largest lesion and all lesions, orientation correction, optional voxel resampling, a saved cache of prepared images, an optional settings search, DCE-MRI phase inputs, LoRA and WiSE-FT fine-tuning options, and a results dashboard at the end. | Settings 5–8, 10, 14 and "Further functions" |

---

## How to start

Open **Section 0, cell 3: Settings, package installation and shared helper functions**. Go through the settings below in order, change the values you need and then choose **Run all** in Google Colab. After you change settings, choose **Runtime > Restart session** and then **Run all** again.

- `True` means enabled and `False` means disabled. Write them without quotation marks.
- A **segmentation**, or **mask**, marks the tumour area in an image.
- Google Colab uses `/content/drive/MyDrive` for the Google Drive folder called **My Drive**.
- To find a setting quickly, search the notebook for its name, for example `DEVELOPMENT_ROOT =`.

**The original (manual) segmentation is always included in the selected classification workflows. nnINTERACTIVE and CAS2 / SAM 2.1 are optional additional resegmentations. They do not replace the original segmentation or its analysis.** You can compare results from the original masks with results from one or both additional mask types.

In this code, **CAS2 uses SAM 2.1**. Its switches therefore contain `SAM2`, while its saved-mask folder is named `CAS2`.

The values shown below are the values currently saved in the notebook. They belong to the last experiment and are not recommended defaults for every dataset. Check them before each new study.

---

## 1. Choose how your data are loaded — the input switch

The notebook can read your data in two ways. **Turn on exactly one of them.** If both are `True`, or both are `False`, the run stops with a clear message.

```python
ENABLE_SPLIT_FOLDERS = True    # Option A: image/mask pairs in development and test folders
ENABLE_NIFTI_PHASES  = False   # Option B: NIfTI images and masks in separate folders
```

| Option | Use it when | Development/test split comes from |
| --- | --- | --- |
| **A. Split folders** (`ENABLE_SPLIT_FOLDERS = True`) | Each patient's image and mask files are stored together, and you have put the development and test patients in separate folders. Usually NRRD files, produced by Code_1. | The folder a patient is in |
| **B. NIfTI folders** (`ENABLE_NIFTI_PHASES = True`) | Your images and masks are NIfTI files (`.nii.gz`) kept in two separate folder trees, as in many public datasets. | A `split` column in the label file |

Each option has its own label file and its own results folder (see below). The notebook picks the correct ones automatically from the switch, so you can keep both blocks filled in and change only the switch.

### Option A — split folders (NRRD image/mask pairs)

```python
DEVELOPMENT_ROOT = [
    "/content/drive/MyDrive/DATA/DEVELOPMENT-165",
    "/content/drive/MyDrive/DATA/none",
]
TEST_ROOT = [
    "/content/drive/MyDrive/DATA/TEST-28",
    "/content/drive/MyDrive/none",
]
SPLIT_FOLDERS_LABELS_CSV = "/content/drive/MyDrive/DATA/labels.csv"
SPLIT_FOLDERS_OUT_ROOT   = "/content/drive/MyDrive/DATA/RESULTS/my-analysis"
```

- **`DEVELOPMENT_ROOT`**: the folder, or list of folders, holding the patients used to develop the models. Several folders are combined into **one development dataset** and read one after the other. Splitting a large dataset over several folders can lower the very high memory use seen when approximately 350 or more files are first read from one folder.
- **`TEST_ROOT`**: **one or two separate test datasets**. These patients are used only to evaluate the final model, never to train it. The final model is trained once on all development patients and then evaluated once on each test dataset. Each test dataset has its own result tables and is named after its folder.
- **`SPLIT_FOLDERS_LABELS_CSV`**: the outcome file for this option (see setting 3).
- **`SPLIT_FOLDERS_OUT_ROOT`**: the results folder for this option (see setting 4).

Rules for both folder settings:

- The code searches the selected folders and all their subfolders.
- Each patient's image file and matching mask file must be in the same folder. NRRD is the usual format, but `.nhdr`, `.nii`, `.nii.gz`, `.mha` and `.mhd` files are also accepted.
- A file is recognised as a mask when its name contains a word such as `seg`, `mask`, `label`, `roi`, `gtv`, `contour`, `annotation` or `delineation`.
- The same patient in more than one folder stops the run.
- A folder path containing `none`, in any mix of upper- and lowercase letters, is skipped with a warning. Use this to switch off an optional folder. Each setting still needs at least one real folder.

### Option B — NIfTI folders (images and masks in separate places)

```python
IMAGE_ROOT             = "/content/drive/MyDrive/MY-DATASET/images"
SEGMENTATION_ROOT      = "/content/drive/MyDrive/MY-DATASET/segmentations"
SEGMENTATION_SUBFOLDER = "expert"   # "expert", "automatic", or None
NIFTI_TUMOR_LABEL      = 1
NIFTI_LABELS_CSV       = "/content/drive/MyDrive/MY-DATASET/labels.csv"
NIFTI_OUT_ROOT         = "/content/drive/MyDrive/MY-DATASET/RESULTS/my-analysis"
```

- **`IMAGE_ROOT`**: the folder with one subfolder per patient. Each patient's images are named like `IMAGE_ROOT/<patient ID>/<patient ID>_0001.nii.gz`, where the last number is the image phase (see "DCE-MRI phases" below).
- **`SEGMENTATION_ROOT`**: the folder with the tumour masks, named `<patient ID>.nii.gz`.
- **`SEGMENTATION_SUBFOLDER`**: the subfolder of `SEGMENTATION_ROOT` that holds the masks you want, for example `"expert"` (manual) or `"automatic"`. Use `None` when the masks are directly in `SEGMENTATION_ROOT`.
- **`NIFTI_TUMOR_LABEL`**: the number that marks tumour in the mask files. Every other number is treated as background.
- **`NIFTI_LABELS_CSV`** and **`NIFTI_OUT_ROOT`**: the label file and results folder for this option.

In this option, `DEVELOPMENT_ROOT` and `TEST_ROOT` are not used. Instead, the label file must contain a `split` column that says which patients are for development (`1`) and which are for testing (`2`). Patient IDs must have the form `PREFIX_number`, for example `PATIENT_001`, and must be identical in the folder names, file names, mask names and label file.

---

## 2. DCE-MRI phases (NIfTI option only)

```python
NIFTI_IMAGE_PHASE = "0001"
DCE_INPUT_MODE    = "off"     # "off", "phases" or "kinetic"
DCE_PRE_PHASE     = "0000"
DCE_LATE_PHASE    = "last"
```

Contrast-enhanced MRI (DCE-MRI) records several images over time: before contrast, early after contrast and late after contrast. These settings choose which of these images the 2D CNNs receive.

| `DCE_INPUT_MODE` | What the CNN receives |
| --- | --- |
| `"off"` | Only the early post-contrast image (`NIFTI_IMAGE_PHASE`). This is the normal setting. |
| `"phases"` | The pre-contrast, early and late images together, so the CNN can see how the tumour takes up contrast. |
| `"kinetic"` | The early image plus how quickly contrast enters (wash-in) and leaves (wash-out) the tumour. |

`DCE_LATE_PHASE = "last"` uses each patient's last available image. Patients missing a needed phase are excluded and listed in an audit file. With Option A (split folders) keep `DCE_INPUT_MODE = "off"`.

---

## 3. Patient outcomes — the label file

This CSV file contains the outcome you want the model to predict. For Option A it is `SPLIT_FOLDERS_LABELS_CSV`, and for Option B it is `NIFTI_LABELS_CSV`.

| Column | Needed for | Meaning |
| --- | --- | --- |
| `patient_id` | Both options | The patient's ID. |
| `label` | Both options | The outcome as `0` or `1`. These two numbers represent the two outcome groups in your study. |
| `split` | Option B only | `1` = development patient, `2` = test patient. |
| `include` | Optional | `1` = use this patient, `0` = leave this patient out of every analysis. Without this column every patient is used. |

For Option A, the simplest file has `patient_id` in column A and the outcome in column B.

Each run uses one label file. To study another outcome, change the label file and also change the results folder, so that earlier results are kept.

---

## 4. Results folder

```python
SPLIT_FOLDERS_OUT_ROOT = "/content/drive/MyDrive/DATA/RESULTS/my-analysis"   # Option A
NIFTI_OUT_ROOT         = "/content/drive/MyDrive/MY-DATASET/RESULTS/my-analysis"   # Option B
```

This is where the code saves result tables, extracted image features (CSV files), trained model files and overview tables. Choose a separate results folder for each analysis you want to keep.

A related setting, `PREPARED_INPUT_CACHE_DIR`, names a Google Drive folder where prepared images are saved. A later session copies them back instead of preparing them again, which saves a lot of time. Use `None` to keep them only for the current session.

---

## 5. Which part of the tumour is analysed

### Largest lesion or all lesions — `ANALYZE_LARGEST_LESION_ONLY`

```python
ANALYZE_LARGEST_LESION_ONLY = True
```

A mask can contain several separate pieces, such as a second tumour or a few stray voxels.

- **`True`**: only the largest piece (by volume) is analysed. The other pieces are counted and logged but ignored.
- **`False`**: all pieces are analysed together. Results go to a separate subfolder called `ALL_LESIONS`, so the two kinds of result never mix.

The choice is made before any resampling, intensity scaling, slice selection or cropping. It applies separately to original, nnINTERACTIVE and CAS2 / SAM 2 masks. The number of pieces and the kept and ignored volumes of each patient are saved in `SHARED_INPUTS_AUDIT/largest_lesion_audit.csv`. Very small stray pieces are also counted, so this is not a clinical tumour count.

### Lesion-size limit — `MAX_LESION_VOLUME_ML`

```python
MAX_LESION_VOLUME_ML = None   # for example 15.0, or None to switch off
```

Leaves out patients whose largest lesion in the original mask is bigger than this volume in millilitres (mL). A lesion of exactly this size is kept. The limit applies to development and test patients and to every workflow. Excluded patients are listed in `SHARED_INPUTS_AUDIT/lesion_volume_filter_audit.csv`.

---

## 6. Image slices used for each patient

### Choose the planes — `SLICE_PLANES`

```python
SLICE_PLANES = ("coronal",)
RUN_SLICE_PLANE_COMPARISON = False
```

With the comparison switched off, `SLICE_PLANES` selects the viewing directions used for each patient. Choose one plane or any combination:

| Example | Directions used |
| --- | --- |
| `("axial",)` | Axial: a horizontal slice through the body. |
| `("sagittal",)` | Sagittal: a side-view slice. |
| `("coronal",)` | Coronal: a front-view slice. |
| `("axial", "coronal")` | Axial and coronal slices. |
| `("axial", "sagittal", "coronal")` | All three planes. |

For each chosen plane, the code takes the slice where the analysed tumour is largest. Keep the comma in a single-plane setting, such as `("coronal",)`.

### Compare several plane choices in one run

```python
RUN_SLICE_PLANE_COMPARISON = True
SLICE_PLANE_COMBINATIONS = (
    ("axial",),
    ("axial", "sagittal", "coronal"),
    ("coronal",),
)
```

The script then tests these three choices in one run, on exactly the same patient splits: **axial only**, **all three planes** and **coronal only**. Replace the entries with the combinations you want to compare. Set `RUN_SLICE_PLANE_COMPARISON = False` to go back to the single choice in `SLICE_PLANES`.

### Compare image size, slice count and augmentation — `RUN_PARAMETER_OPTIMIZATION`

```python
RUN_PARAMETER_OPTIMIZATION = False
OPTIMIZATION_INPUT_SIZES = (128, 224)
OPTIMIZATION_AUG_LEVELS  = ("FLIPS", "BASIC", "COMPLEX")
EVALUATE_HOLDOUT_FOR_ALL_COMPARISON_ARMS = True
```

`True` tests every combination of the listed image sizes (in pixels) and training augmentation levels (how much the training images are randomly flipped, rotated or zoomed). Each tested combination is compared with a reference setting on the same patients. With `EVALUATE_HOLDOUT_FOR_ALL_COMPARISON_ARMS = False`, choices are compared on development data only and the test patients stay unseen. `CNN_INPUT_SIZE_SHARED` (128 or 224) sets the image size when this comparison is off.

---

## 7. Image orientation and voxel size

```python
USE_LPS_REORIENTATION  = True
USE_OBLIQUE_CORRECTION = True
CNN_VOXEL_MODE         = "original"         # "original" or "resample"
CNN_RESAMPLE_SPACING_MM = (1.0, 1.0, 1.0)
```

- **`USE_LPS_REORIENTATION`**: `True` turns every image into the same standard orientation, so "axial", "sagittal" and "coronal" mean the same anatomical direction for every patient. Image values are not changed. `False` keeps the orientation stored in the file.
- **`USE_OBLIQUE_CORRECTION`**: `True` straightens images that were acquired at a tilt. `False` keeps the tilt.
- **`CNN_VOXEL_MODE`**: `"original"` keeps each scan's own voxel size. `"resample"` converts all scans to the voxel size in `CNN_RESAMPLE_SPACING_MM` (in millimetres). Resampled results are written to a separate subfolder.

These settings change only the images given to the classification models. The automatic segmentations and their Dice comparisons always use the image as stored. Source files are never overwritten.

---

## 8. Analysis workflows — three independent switches

```python
RUN_FROZEN_CNN_E2E   = True
RUN_CNN_LR           = False
RUN_CNN_FINETUNE_E2E = True
```

Switch on the workflows you want to compare. A CNN is a neural network that learns from images. Its **backbone** is the main part that extracts image information.

| Setting | What it enables |
| --- | --- |
| `RUN_FROZEN_CNN_E2E` | Section 1, `frozen`: trains a classifier while keeping the pretrained backbone unchanged. |
| `RUN_CNN_LR` | Section 2, `efficient_ML`: extracts image features with an unchanged pretrained backbone, then uses them in classical machine-learning models. The models are chosen in `RUN_CLASSIFIERS`, for example `['NAIVE_BAYES', 'LOGREG']`. |
| `RUN_CNN_FINETUNE_E2E` | Section 3, `finetune`: trains a classifier and also adjusts the pretrained backbone using the development data. Needs a GPU. |

`False` skips that workflow. These choices apply to the original-mask analysis and to any additional mask analyses you switch on.

`FROZEN_E2E_FINAL_MODEL` chooses how Section 1 scores the test patients: `"refit"` (default; one model retrained on all development patients) or `"fold_ensemble"` (the average of the cross-validation models).

---

## 9. CNN models — `SHARED_BACKBONES`

```python
SHARED_BACKBONES = [
#   "radimagenet_densenet121",
#   "imagenet_densenet121",
#   "radimagenet_inceptionv3",
#   "imagenet_inceptionv3",
    "radimagenet_resnet10t",
#   "imagenet_resnet10t",
#   "radimagenet_resnet18",
#   "imagenet_resnet18",
#   "radimagenet_resnet50",
#   "imagenet_resnet50",
#   "dinov3_vits16",
#   "dinov3_vitb16",
#   "mars_3d",
#   "choi_idh",
]
```

This list selects the CNN models used in the three workflows. A line beginning with `#` is inactive. Remove the `#` to include a model and add it to leave a model out.

| Model name | Pretrained on | Workflows |
| --- | --- | --- |
| `radimagenet_resnet10t`, `_resnet18`, `_resnet50`, `_densenet121` | RadImageNet (radiology images) | All three |
| `imagenet_resnet10t`, `_resnet18`, `_resnet50`, `_densenet121` | ImageNet (everyday photographs) | All three |
| `radimagenet_inceptionv3`, `imagenet_inceptionv3` | RadImageNet / ImageNet | Section 2 only |
| **`dinov3_vits16`** (small) and **`dinov3_vitb16`** (base) — **new** | DINOv3: a Vision Transformer from Meta, trained without labels on 1.7 billion photographs | All three |
| **`mars_3d`** — **new** | MARS: a 3D model pretrained on MRI | All three |
| `choi_idh` | A published brain-tumour (IDH) network | Section 2 only |

The same architecture name with a different prefix (for example `radimagenet_resnet10t` and `imagenet_resnet10t`) is the same network with different pretrained weights. Sections 1 and 3 report and skip models they do not support.

**About MARS.** Unlike the other models, which look at 2D slices, MARS looks at a 3D block of the image around the whole lesion. The plane and slice settings therefore do not affect MARS. Its main settings are `MARS_INPUT_SIZE` (block size, default 96 × 96 × 96) and `MARS_CROP_MARGIN_MM` (extra tissue around the lesion, default 5 mm). Its official weights are downloaded once to `MARS_WEIGHTS_PATH` and checked automatically.

**About DINOv3.** Its weights are downloaded once from Hugging Face. `HF_TOKEN` is an optional Hugging Face access key and can normally stay empty.

Weights for the RadImageNet models must be downloaded by you (see "CNN weights" below).

---

## 10. Fine-tuning options (Section 3)

All Section 3 settings are in the block `FINE-TUNING SETTINGS (Section 3)` just below the main settings. The most useful new options are:

| Setting | What it does |
| --- | --- |
| `FINETUNE_TRAINABLE` | How much of each backbone is adjusted: `"auto"` (a small built-in part), a number of blocks such as `3`, a share such as `"50%"`, or `"all"`. |
| `USE_LORA` | LoRA adjusts a small number of extra weights instead of the original ones. This lowers the risk of overfitting a small dataset. Off by default. `LORA_BACKBONES` chooses the model families it applies to. |
| `USE_WISE_FT`, `WISE_FT_ALPHA` | WiSE-FT mixes the fine-tuned model with the model from before fine-tuning. This often makes results more stable on data from another hospital, at no extra training cost. |
| `USE_MOVEMENT_CAP` | Stops fine-tuning when the backbone moves too far (default 3%) from its pretrained weights, so it keeps what it learned during pretraining. |

Runs with non-default fine-tuning settings write to their own results folders, so they never overwrite the default results.

---

## 11. Additional segmentations and mask comparison (nnINTERACTIVE and SAM 2)

```python
RUN_AUTOSEG_NNINTERACTIVE = True
RUN_AUTOSEG_SAM2          = False

ANALYZE_NNINTERACTIVE_MASKS = True
ANALYZE_SAM2_MASKS          = False
```

**Purpose.** These functions answer two questions:

1. **How closely does an automatic segmentation agree with the manual one?** This shows how much the tumour outline varies between a human and an automatic method (manual-versus-automatic, or inter-observer, variability).
2. **Do the classification results change when automatic masks are used instead of manual masks?**

The code can create additional tumour masks with **nnINTERACTIVE** and **CAS2 / SAM 2.1**. It compares each new mask with the original mask using the **Dice Similarity Coefficient (DSC)**. A DSC of `1` means the masks match exactly and `0` means they do not overlap at all. DSC measures agreement with the original mask. It does not prove that either mask is correct.

The existing masks provide the guidance, called **prompts**, for creating the new masks. These are therefore guided resegmentations, not segmentations made independently of the original masks.

- **nnINTERACTIVE** works with the whole 3D image. It gets one prompt, on the slice where each tumour is largest, and draws the full 3D mask itself.
- **CAS2 / SAM 2.1** works slice by slice in 2D. It gets one rectangle (box) around the tumour on every slice containing tumour. Because these boxes already come from the manual mask, part of its agreement is built in. The column `dice_box_only_control` shows how much: it is the DSC of a simple oval drawn inside the same boxes without looking at the image.

Creating masks and analysing them have separate switches:

| Setting | What it enables |
| --- | --- |
| `RUN_AUTOSEG_NNINTERACTIVE` | Creates or loads nnINTERACTIVE masks and compares them with the original masks. |
| `RUN_AUTOSEG_SAM2` | Creates or loads CAS2 / SAM 2.1 masks and compares them with the original masks. |
| `ANALYZE_NNINTERACTIVE_MASKS` | Repeats the selected classification workflows using the nnINTERACTIVE masks. |
| `ANALYZE_SAM2_MASKS` | Repeats the selected classification workflows using the CAS2 / SAM 2.1 masks. |

- To create **and** analyse a mask type, set both of its switches to `True`.
- To only compare masks (DSC) without extra classification analyses, switch on its `RUN_AUTOSEG` switch and switch off its `ANALYZE` switch.
- To use masks saved in an earlier run without creating new ones, switch off `RUN_AUTOSEG` and switch on `ANALYZE`.
- With both `RUN_AUTOSEG` switches `False`, the segmentation cell is skipped and its extra packages are not installed. Keep both `ANALYZE` switches `False` for the original-mask analysis only.

**With the settings shown above, classification uses both the original and nnINTERACTIVE masks, and CAS2 is off. To include all three mask types, set all four switches to `True`. The original-mask analysis is always included.**

**Results.** The agreement tables and overlay pictures are saved in `OUT_ROOT/AUTO_SEGMENTATION_CHECK`:

- `auto_segmentation_dice_per_patient.csv`: one row per patient and method, with DSC, DSC of the largest lesion, tumour volumes and whether the same slice was chosen for analysis.
- `auto_segmentation_dice_summary.csv`: median, quartiles, mean and minimum DSC per method, and how many patients fell below `AUTOSEG_LOW_DICE` (default 0.70).
- Overlay figures showing the manual and automatic outlines for a few patients.

Classification results with the extra masks are saved under `OUT_ROOT/SEGMENTATION_ANALYSES/`. The main overview table `combined_pipeline_overview.csv` then contains all mask types, marked in the `segmentation_method` column (`original`, `sam2_box_per_slice` or `nninteractive_<prompt type>`).

**Model weights.** Set `NNINTERACTIVE_WEIGHTS_PATH` and `SAM2_WEIGHTS_PATH` to the folders holding the downloaded weights, or set `AUTOSEG_DOWNLOAD_MISSING_WEIGHTS = True` to let the notebook download them once:

- nnINTERACTIVE v1.0 (about 410 MB, non-commercial licence CC BY-NC-SA 4.0): https://huggingface.co/MIC-DKFZ/nnInteractive/tree/main/nnInteractive_v1.0
- SAM 2.1 Hiera Base+ (about 325 MB, Apache-2.0 licence): https://huggingface.co/facebook/sam2.1-hiera-base-plus/tree/main

**Test on a few patients first.** In Section 0, cell 5, `AUTOSEG_PILOT_N_PATIENTS` can be set to a small number to try the segmentation on that many development patients only. Set it back to `None` for a full analysis.

### nnINTERACTIVE prompt type — `NNINTERACTIVE_ANALYSIS_PROMPT_TYPE`

```python
NNINTERACTIVE_ANALYSIS_PROMPT_TYPE = "lasso"
```

Chooses the kind of guidance nnINTERACTIVE receives. Only the chosen type is created, compared and analysed in a run.

| Value | Guidance given on the slice where the tumour is largest |
| --- | --- |
| `"lasso"` (default) | A filled outline drawn around the tumour. |
| `"box"` | A tight rectangle around the tumour. |
| `"scribble"` | A short stroke inside the tumour. |
| `"point"` | One point well inside the tumour. |

---

## 12. Saved segmentation folders

```python
CAS2_ROOT     = "/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/CAS2"
LASSO_ROOT    = "/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/nninteraction-lasso"
BOX_ROOT      = "/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/nninteraction-box"
SCRIBBLE_ROOT = "/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/nninteraction-scribble"
POINT_ROOT    = "/content/drive/MyDrive/DATA/AUTOMATIC-SEG-FILES/nninteraction-point"
```

New masks are saved permanently as one NRRD file per patient in Google Drive. Each mask type has its own folder:

| Setting | Mask type |
| --- | --- |
| `CAS2_ROOT` | CAS2 / SAM 2.1 |
| `LASSO_ROOT` | nnINTERACTIVE with lasso guidance |
| `BOX_ROOT` | nnINTERACTIVE with box guidance |
| `SCRIBBLE_ROOT` | nnINTERACTIVE with scribble guidance |
| `POINT_ROOT` | nnINTERACTIVE with point guidance |

Before creating a mask, the code checks whether it already exists. Existing masks are loaded and reused, and only missing masks are created and saved. Keeping the same folders lets later runs reuse the masks and saves processing time. To create a patient's mask again, delete or rename that patient's saved file first.

These folders store the **additional masks**. The results folder stores the **analysis results**. The original image and mask files stay where they are.

---

## 13. Contour perturbation test — robustness to changes in the tumour outline (Section 7)

```python
RUN_MASK_PERTURBATION_RETEST = True
PERTURB_RETEST_ARMS = ("finetune_cnn",)
PERTURB_RETEST_LEVELS = (
    ("dilate", 1), ("dilate", 2), ("dilate", 3),
    ("erode", 1), ("erode", 2), ("erode", 3),
    ("shift", 2.0), ("shift", 5.0),
    ("scale", 0.1), ("scale", -0.1), ("scale", 0.2), ("scale", -0.2),
)
```

**Purpose.** The tumour mask decides which slice is analysed, where the image is cropped and how image brightness is scaled. Two readers rarely draw exactly the same outline. This test answers the question: **"How much would the result change if the tumour outline (the border of the volume of interest, VOI) had been drawn slightly differently?"** It also shows how accurate an automatic segmentation would need to be to replace manual outlining.

**How it works.** No model is retrained. The models saved by an earlier full run are reloaded and kept unchanged. Only the tumour mask of each **test** patient is changed, and the test patients are scored again. Any change in the result therefore comes from the outline alone.

**Ways of changing the outline** (`PERTURB_RETEST_LEVELS`):

| Change | Meaning | Size is given in |
| --- | --- | --- |
| `"dilate"` | Makes the outline larger on all sides. | Voxels, for example `1` |
| `"erode"` | Makes the outline smaller on all sides. | Voxels |
| `"shift"` | Moves the whole outline in a random direction. | Millimetres, for example `2.0` |
| `"scale"` | Makes the outline larger or smaller by a percentage. | A fraction: `0.2` = 20% larger, `-0.1` = 10% smaller |

**Other settings:**

| Setting | What it does |
| --- | --- |
| `PERTURB_RETEST_ARMS` | Which saved models to test: `"finetune_cnn"` (Section 3 fine-tuned CNN), `"frozen_ml"` (Section 2 features plus machine learning) or both. Keep the comma for one entry, for example `("finetune_cnn",)`. |
| `PERTURB_RETEST_CNN_MODEL_PATH`, `PERTURB_RETEST_ML_MODEL_PATH` | The saved model files to test. `None` searches the results folder and stops with a list if it finds more than one match. |
| `PERTURB_RETEST_BACKBONE` | The CNN model name of the saved model, for example `"radimagenet_resnet10t"`. |
| `PERTURB_RETEST_TEST_SET` | Which test dataset to use, named after its `TEST_ROOT` folder, for example `"TEST-28"`. |
| `PERTURB_RETEST_LABEL_TAG` | Which outcome table to use. `None` uses the first one. |
| `PERTURB_RETEST_SHIFT_REPEATS` | How many random shifts are tried per patient (default 5). The other changes are not random and are done once. |
| `PERTURB_RETEST_SEED` | Fixes the random shifts, so a rerun gives exactly the same changed masks. |
| `PERTURB_RETEST_MASK_DIR` | Where the changed masks are saved and reused. `None` uses `OUT_ROOT/MASK_PERTURBATION`. |

**Before you use it.** A normal full run must have finished first, because this test reloads its saved models. You can then set `RUN_CNN_FINETUNE_E2E = False` and rerun to repeat only the perturbation test. Nothing is retrained and no GPU is needed.

**Results** (in `OUT_ROOT/MASK_PERTURBATION_RETEST`):

- `perturbation_summary.csv`: one row per change, with
  - `dsc_median`: how well the changed outline agrees with the original one. Published inter-observer DSC values for meningioma are about 0.85 to 0.93, so rows in that range represent realistic disagreement between human readers.
  - `same_central_slice_pct`: how often the same slice is still chosen for analysis.
  - `auc_mean` and `auc_sd`: the test-set AUC and its spread over the random shifts.
  - `auc_delta_vs_original` with `delta_ci_low` and `delta_ci_high`: the AUC change compared with the original outline, with its confidence interval. An interval that contains 0 means the change did not measurably affect the result.
- `perturbation_per_pass.csv`: one row per model, change and random draw.
- `auc_against_dsc.png`: a figure of AUC against DSC. The DSC at which the AUC starts to fall is the quality an automatic segmentation would have to reach.

`False` skips this test and changes nothing else in the notebook.

---

## 14. Results at the end of a run

- **`combined_pipeline_overview.csv`** (in `COMBINED_OVERVIEW_DIR`) is the main result table. It lists every model of every enabled workflow and mask type, with development (cross-validation) and test-set results. The `workflow` column shows `frozen`, `efficient_ML` or `finetune`. With two test datasets, one extra file is written per test dataset.
- **Results dashboard.** The last cell repeats every result table and saves `all_registered_result_tables_long.csv` (all tables combined), an index of these tables and an index of every CSV file written during the run.
- **Audit files** in `OUT_ROOT/SHARED_INPUTS_AUDIT` list patients that were excluded (for example missing masks or labels, lesions above the size limit) and the lesion counts.
- AUC values are reported with confidence intervals. `TOP_RISK_PERCENT` also reports results for the patients with the highest predicted risk.

---

## Repository contents

This repository consists of two main parts:

- **Code_1 (DICOM or MRB to NRRD conversion):** a script that converts raw DICOM or MRB data into paired NRRD image and mask files.
- **Code_2 (analysis pipeline):** the main notebook. It reads either the NRRD image/mask pairs created by Code_1 (Option A) or NIfTI image and mask folders (Option B).

## CNN weights

Download the pretrained RadImageNet weights for the CNN models you use and place the files in the folder set by `WEIGHTS_BASE`:

```python
WEIGHTS_BASE = "/content/drive/MyDrive/DATA/weights"
```

You can change `WEIGHTS_BASE` in Code_2 to any folder in your Google Drive.

- DenseNet121: https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/DenseNet121.pt?download=true
- ResNet50: https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt?download=true
- ResNet18: https://huggingface.co/convergedmachine/RadImagenet/resolve/main/resnet18.pth?download=true
- ResNet10t: https://huggingface.co/convergedmachine/RadImagenet/resolve/main/resnet10t.pth?download=true
- InceptionV3: https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/InceptionV3.pt?download=true

These weights are downloaded automatically and need no manual step:

- ImageNet models: downloaded by the standard Python model libraries.
- DINOv3 (from Hugging Face): https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m
- MARS (official weights, downloaded once to `MARS_WEIGHTS_PATH` when `MARS_DOWNLOAD_WEIGHTS = True`): https://github.com/zqiuak/MARS
- nnINTERACTIVE and SAM 2.1: see setting 11.

## Further instructions and settings

Detailed instructions for every setting are given in the first notebook cell (Section 0, cell 1) and in the comments within the code itself.

## How to cite

If you use this code in a published study, please cite it to ensure proper academic credit. You can use the "Cite this repository" button on the right side of the GitHub page or the following reference:

(submitted for publication)
[name/authors]. (2026). [Repository Name]. GitHub. [GitHub URL]
