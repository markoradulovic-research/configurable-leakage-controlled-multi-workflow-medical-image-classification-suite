# configurable-leakage-controlled-multi-workflow-medical-image-classification-suite

A configurable Google Colab notebook that compares leakage-controlled deep-learning and machine-learning workflows for medical image classification. "Leakage-controlled" means that the test patients are never used for training or for choosing settings. Everything runs in one notebook and the main settings are grouped at the top. Researchers who do not write code can use it, and developers can change every part of it.

---

## Overview of the code

### Workflows

Each workflow is switched on or off on its own. All enabled workflows use the same patients, the same split and the same prepared images, so their results are directly comparable.

| Workflow | Switch | What it does | Trains the backbone |
| --- | --- | --- | --- |
| Section 1, `frozen` | `RUN_FROZEN_CNN_E2E` | Trains a classification head on top of an unchanged pretrained network. | No |
| Section 2, `efficient_ML` | `RUN_CNN_LR` | Extracts features with an unchanged pretrained network and radiomics, then fits classical machine-learning models on them. | No |
| Section 3, `finetune` | `RUN_CNN_FINETUNE_E2E` | Trains the head and also adjusts the pretrained network. Needs a GPU. | Yes |
| Section 3b, token aggregator | `RUN_TOKEN_AGGREGATOR` | Stores the internal 3D feature grid (tokens) of each patient once and trains a small transformer on it. Compared with a ridge model on the same folds. | No |

### Backbones (`SHARED_BACKBONES`)

| Name | Pretrained on | Input | Sections 1 and 3 | Section 2 |
| --- | --- | --- | --- | --- |
| `radimagenet_resnet10t`, `radimagenet_resnet18`, `radimagenet_resnet50`, `radimagenet_densenet121` | RadImageNet, radiology images | 2D slices | Yes | Yes |
| `imagenet_resnet10t`, `imagenet_resnet18`, `imagenet_resnet50`, `imagenet_densenet121` | ImageNet, everyday photographs | 2D slices | Yes | Yes |
| `radimagenet_inceptionv3`, `imagenet_inceptionv3` | RadImageNet or ImageNet | 2D slices | No, features only | Yes |
| `dinov3_vits16`, `dinov3_vitb16` | DINOv3 Vision Transformers, trained without labels | 2D slices | Yes | Yes |
| `convnext_tiny_dino3`, `convnextv2_atto`, `convnextv2_nano`, `convnextv2_tiny` | DINOv3 ConvNeXt and ConvNeXt-V2 | 2D slices | Yes | Yes |
| `mars_3d` | MARS, pretrained on MRI | 3D block | Yes | Yes |
| `fmcib_3d` | FMCIB, pretrained on CT lesions | 3D block | Yes | Yes |
| `choi_idh` | Published brain-tumour (IDH) network | 2D slices | No, features only | Yes |

The same architecture with another prefix, for example `radimagenet_resnet10t` and `imagenet_resnet10t`, is the same network with different pretrained weights. Sections 1 and 3 report and skip any backbone they cannot train.

### Preprocessing options

| Option | Setting | Choices |
| --- | --- | --- |
| Lesion analysed | `ANALYZE_LARGEST_LESION_ONLY` | Largest piece of the mask, or all pieces together |
| Lesion-size limit | `MAX_LESION_VOLUME_ML` | Exclude patients above a volume in millilitres, or `None` |
| Viewing planes | `SLICE_PLANES` | Any combination of axial, sagittal and coronal |
| Image size | `CNN_INPUT_SIZE_SHARED` | 128 or 224 pixels |
| Orientation | `USE_LPS_REORIENTATION`, `USE_OBLIQUE_CORRECTION` | Standard orientation and straightening of tilted scans, each on or off |
| Voxel size | `CNN_VOXEL_MODE`, `CNN_RESAMPLE_SPACING_MM` | Keep each scan's own voxel size, or resample all scans to one size |
| Bias-field correction | `USE_N4_BIAS_CORRECTION` | N4 correction of slow shading, on or off |
| Intensity scaling | `INTENSITY_NORMALIZATION_MODE` | Whole-image z-score, whole-image percentiles or z-score inside the tumour |
| Crop | `CROP_MIN_FOV_MM` | Tight square crop around the tumour, or a crop of at least a given width in millimetres so that tumour size stays visible |
| Surroundings | `TUMOR_CONTEXT_MODE` | Keep the tissue around the tumour, or blank it out |
| Input channels | `FRONTEND_MODE`, `FRONTEND_CHANNELS` | Grey image in all 3 channels, or fixed texture, edge, rank, window and contrast-phase channels |
| Contrast phases (2D) | `DCE_INPUT_MODE` | Early phase only, 3 phases or early phase with wash-in and wash-out |
| Contrast phases (3D) | `VOLUME_DCE_PHASES` | Early phase only, or pre-contrast, early and late |
| Training augmentation | `TRAIN_AUGMENTATION_LEVEL` | `FLIPS`, `BASIC`, `COMPLEX` or `COMPLEX_JITTER` |

### Feature extraction

| Feature type | Where it comes from | Used by |
| --- | --- | --- |
| CNN features | Each backbone runs with frozen weights over the chosen planes. Slice features are averaged into 1 vector per patient. 3D backbones give 1 vector per volume. | Section 2 |
| Global and local 3D features | `MARS_USE_GLOBAL_LOCAL`, `FMCIB_USE_GLOBAL_LOCAL`. The whole lesion plus 7 smaller views inside it, through 1 shared network. | Sections 1, 2 and 3 |
| Radiomics features | PyRadiomics, computed in 3D on the native grid with `RUN_RADIOMICS`. Shape, intensity and texture classes, with Laplacian-of-Gaussian filters. | Section 2 |
| CNN plus radiomics | 1 combined table per CNN feature table (early fusion). | Section 2 |
| Late fusion | The best CNN model and the best radiomics model, averaged after standardisation. | Section 2 |
| Lesion tokens | The full 3D feature grid of each patient, stored once. | Section 3b |

### Optional analyses

| Analysis | Switch | Purpose |
| --- | --- | --- |
| Additional segmentations | `RUN_AUTOSEG_NNINTERACTIVE`, `RUN_AUTOSEG_SAM2` | Create automatic tumour masks and measure their agreement with the manual masks. |
| Classification on those masks | `ANALYZE_NNINTERACTIVE_MASKS`, `ANALYZE_SAM2_MASKS` | Repeat the whole analysis on the automatic masks. |
| Setting comparison | `RUN_PARAMETER_OPTIMIZATION`, `RUN_SLICE_PLANE_COMPARISON` | Compare image sizes, augmentation levels and plane choices on identical folds. |
| Contour perturbation test | `RUN_MASK_PERTURBATION_RETEST` | Measure how much the result changes when the tumour outline is drawn differently. |
| Calibration and decision curves | `RUN_CALIBRATION_AND_DECISION_CURVES` | Check whether predicted risks are realistic and clinically useful. |
| Saliency check | `RUN_SALIENCY_CHECK` | Check how much of the prediction comes from the tumour itself. |

---

## How to start

Open **Section 0, cell 3, Settings, package installation and shared helper functions**. Go through the settings below in order, change the values you need and then choose **Run all** in Google Colab. After changing settings, choose **Runtime > Restart session** and then **Run all** again.

- `True` means enabled and `False` means disabled. Write them without quotation marks.
- A few settings only work together with another one, for example the contrast-phase input, which needs the NIfTI layout. The notebook resolves those pairs by itself before anything runs and lists every change it made under `[AUTO SETTINGS]`.
- A **segmentation**, or **mask**, marks the tumour area in an image.
- Google Colab uses `/content/drive/MyDrive` for the Google Drive folder called **My Drive**.
- To find a setting quickly, search the notebook for its name, for example `DEVELOPMENT_ROOT =`.
- The values saved in the notebook come from the last experiment. They are examples, not recommended defaults. Check them before each study.

**The original manual segmentation is always included in the selected workflows. nnINTERACTIVE and CAS2 / SAM 2.1 are optional extra resegmentations that do not replace it.** In this code, **CAS2 uses SAM 2.1**, so its switches contain `SAM2` while its saved-mask folder is named `CAS2`.

---

## 1. How your data are loaded

**Turn on exactly one of the two layouts.** If both are `True`, or both are `False`, the run stops with a clear message.

```python
ENABLE_SPLIT_FOLDERS = True    # Option A: image/mask pairs in development and test folders
ENABLE_NIFTI_PHASES  = False   # Option B: NIfTI images and masks in separate folders
```

| Option | Use it when | Development and test split comes from |
| --- | --- | --- |
| **A. Split folders** | Each patient's image and mask files are stored together and the development and test patients are in separate folders. Usually NRRD files produced by Code_1. | The folder a patient is in |
| **B. NIfTI folders** | Images and masks are NIfTI files kept in 2 separate folder trees, as in many public datasets. | A `split` column in the label file |

Each option has its own label file and its own results folder. The notebook picks the correct ones from the switch, so both blocks can stay filled in.

### Option A, split folders

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

- **`DEVELOPMENT_ROOT`**: the folder, or list of folders, holding the patients used to develop the models. Several folders are combined into **one development dataset** and read one after the other. **For large datasets, split the development patients over several folders.** Reading roughly 350 files or more from one Google Drive folder at the start of a run can briefly use over 50 GB of memory and crash the Colab session. Reading several smaller folders avoids this. The results are the same.
- **`TEST_ROOT`**: 1 or 2 separate test datasets, used only to evaluate the final model. The final model is trained once on all development patients and evaluated once on each test dataset, with its own result tables named after its folder.

Rules for both folder settings:

- The selected folders and all their subfolders are searched.
- Each patient's image file and matching mask file must be in the same folder. Accepted formats are `.nrrd`, `.nhdr`, `.nii`, `.nii.gz`, `.mha` and `.mhd`.
- A file counts as a mask when its name contains a word such as `seg`, `mask`, `label`, `roi`, `gtv`, `contour`, `annotation` or `delineation`.
- The same patient in more than 1 folder stops the run.
- A folder path containing `none`, in any mix of upper and lower case, is skipped with a warning. Use this to switch off an optional folder. Each setting still needs at least 1 real folder.

### Option B, NIfTI folders

```python
IMAGE_ROOT             = "/content/drive/MyDrive/MY-DATASET/images"
SEGMENTATION_ROOT      = "/content/drive/MyDrive/MY-DATASET/segmentations"
SEGMENTATION_SUBFOLDER = "expert"   # "expert", "automatic", or None
NIFTI_TUMOR_LABEL      = 1
NIFTI_LABELS_CSV       = "/content/drive/MyDrive/MY-DATASET/labels.csv"
NIFTI_OUT_ROOT         = "/content/drive/MyDrive/MY-DATASET/RESULTS/my-analysis"
```

- **`IMAGE_ROOT`**: 1 subfolder per patient, with images named `IMAGE_ROOT/<patient ID>/<patient ID>_0001.nii.gz`, where the last number is the contrast phase.
- **`SEGMENTATION_ROOT`**: the tumour masks, named `<patient ID>.nii.gz`.
- **`SEGMENTATION_SUBFOLDER`**: the subfolder holding the masks you want, for example `"expert"` or `"automatic"`. Use `None` when the masks are directly in `SEGMENTATION_ROOT`.
- **`NIFTI_TUMOR_LABEL`**: the number that marks tumour in the mask files. Every other number is background.

Here `DEVELOPMENT_ROOT` and `TEST_ROOT` are not used. The label file must contain a `split` column, with `1` for development and `2` for test. Patient IDs must have the form `PREFIX_number`, for example `PATIENT_001`, and must be identical in the folder names, file names, mask names and label file.

---

## 2. Contrast-enhanced MRI phases

Contrast-enhanced MRI (DCE-MRI) records several images over time, before contrast, early after contrast and late after contrast. These settings choose which images the networks receive. They need the NIfTI layout. With split folders, keep `DCE_INPUT_MODE = "off"`.

```python
NIFTI_IMAGE_PHASE  = "0001"
DCE_INPUT_MODE     = "off"     # "off", "phases" or "kinetic"
DCE_PRE_PHASE      = "0000"
DCE_LATE_PHASE     = "last"
VOLUME_DCE_PHASES  = ("pre", "early", "late")
```

| `DCE_INPUT_MODE` | What the 2D networks receive |
| --- | --- |
| `"off"` | Only the early post-contrast image. This is the normal setting. |
| `"phases"` | Pre-contrast, early and late images together, on 1 shared scale, so the networks can see how the tumour takes up contrast. |
| `"kinetic"` | The early image plus how quickly contrast enters (wash-in) and leaves (wash-out) the tumour. |

`VOLUME_DCE_PHASES` does the same for the 3D backbones MARS and FMCIB. With 3 phases, each patient is described by the early phase plus the differences between phases. `DCE_LATE_PHASE = "last"` uses each patient's last available image. Patients missing a needed phase are excluded and listed in an audit file.

---

## 3. Patient outcomes, the label file

This CSV file contains the outcome the model should predict.

| Column | Needed for | Meaning |
| --- | --- | --- |
| `patient_id` | Both options | The patient's ID. |
| `label` | Both options | The outcome as `0` or `1`, the two outcome groups of your study. |
| `split` | Option B only | `1` = development patient, `2` = test patient. |
| `include` | Optional | `1` = use this patient, `0` = leave this patient out of every analysis. Without this column every patient is used. |

For Option A, the simplest file has `patient_id` in column A and the outcome in column B. Each run uses 1 label file. To study another outcome, change the label file and also the results folder, so that earlier results are kept.

---

## 4. Results folder and prepared-image cache

```python
SPLIT_FOLDERS_OUT_ROOT   = "/content/drive/MyDrive/DATA/RESULTS/my-analysis"           # Option A
NIFTI_OUT_ROOT           = "/content/drive/MyDrive/MY-DATASET/RESULTS/my-analysis"     # Option B
PREPARED_INPUT_CACHE_DIR = "/content/drive/MyDrive/MY-DATASET/PREPARED_INPUT_CACHE"
```

The results folder receives result tables, feature files, trained models and overview tables. Choose a separate folder for each analysis you want to keep.

`PREPARED_INPUT_CACHE_DIR` stores the prepared images on Google Drive. A later session copies them back instead of preparing them again, which saves a lot of time. Each cached file is tied to its source files and to every preprocessing setting, so a changed setting prepares only the images it affects. Use `None` to keep the prepared images for the current session only.

---

## 5. Which part of the tumour is analysed

```python
ANALYZE_LARGEST_LESION_ONLY = True
MAX_LESION_VOLUME_ML        = None   # for example 15.0, or None to switch off
```

A mask can contain several separate pieces, such as a second tumour or a few stray voxels.

- **`True`**: only the largest piece by volume is analysed. The other pieces are counted and logged but ignored.
- **`False`**: all pieces are analysed together. Results go to a separate subfolder called `ALL_LESIONS`, so the two kinds of result never mix.

The choice is made before any resampling, intensity scaling, slice selection or cropping, and applies to the original and the automatic masks alike. Piece counts and volumes per patient are saved in `SHARED_INPUTS_AUDIT/largest_lesion_audit.csv`. Very small stray pieces are counted too, so this is not a clinical tumour count.

`MAX_LESION_VOLUME_ML` leaves out patients whose largest lesion in the original mask is bigger than this volume in millilitres. A lesion of exactly this size is kept. The limit applies to development and test patients and to every workflow. Excluded patients are listed in `SHARED_INPUTS_AUDIT/lesion_volume_filter_audit.csv`.

---

## 6. Image slices and crop

```python
SLICE_PLANES          = ("coronal",)
CNN_INPUT_SIZE_SHARED = 224     # 128 or 224 pixels
CROP_MIN_FOV_MM       = None    # for example 80
TUMOR_CONTEXT_MODE    = "MIN_BBOX"   # "MIN_BBOX" or "ROI_ONLY"
```

`SLICE_PLANES` selects the viewing directions used for each patient. Choose 1 plane or any combination of `"axial"` (horizontal slice), `"sagittal"` (side view) and `"coronal"` (front view). For each chosen plane the code takes the slice where the analysed tumour is largest. Keep the comma in a single-plane setting, such as `("coronal",)`. These settings do not affect the 3D backbones, which read a block around the whole lesion.

Each plane gets its own square crop around the tumour, widened by a small safety margin, and is resized to `CNN_INPUT_SIZE_SHARED` pixels. The whole tumour of that plane always lies inside the crop.

- **`CROP_MIN_FOV_MM = None`**: every tumour is cropped tightly and stretched to fill the image, so tumour size is not visible to the network.
- **A number of millimetres**: the crop is at least that wide and centred on the tumour, so smaller tumours keep their true relative size. Tumours wider than the value keep their tight crop, and the code reports how many there are.
- **`TUMOR_CONTEXT_MODE`**: `"MIN_BBOX"` keeps the tissue around the tumour inside the crop. `"ROI_ONLY"` replaces everything outside the tumour outline with a flat value, so only the tumour itself is seen.

A visual check of the analysed slices is printed for `QC_N_PATIENTS` patients when `SHOW_ANALYSED_SLICES = True`.

---

## 7. Image orientation and voxel size

```python
USE_LPS_REORIENTATION   = True
USE_OBLIQUE_CORRECTION  = True
CNN_VOXEL_MODE          = "original"         # "original" or "resample"
CNN_RESAMPLE_SPACING_MM = (1.0, 1.0, 1.0)
```

- **`USE_LPS_REORIENTATION`**: `True` turns every image into the same standard orientation, so the plane names mean the same anatomical direction for every patient. Image values are not changed.
- **`USE_OBLIQUE_CORRECTION`**: `True` straightens images acquired at a tilt.
- **`CNN_VOXEL_MODE`**: `"original"` keeps each scan's own voxel size. `"resample"` converts all scans to `CNN_RESAMPLE_SPACING_MM` in millimetres and writes to a separate subfolder.

These settings change only the images given to the classification models. Automatic segmentation and its agreement measurements always use the image as stored, and source files are never overwritten.

---

## 8. Intensity scaling and input channels

```python
INTENSITY_NORMALIZATION_MODE = "image_zscore"   # "image_zscore", "image_percentile" or "tumor_zscore"
USE_N4_BIAS_CORRECTION       = False
FRONTEND_MODE                = "off"            # "off" or "on"
FRONTEND_CHANNELS            = ("intensity", "energy:1.5", "energy:3.0")
```

MRI has no absolute intensity scale, so every image is rescaled before it reaches a network.

| `INTENSITY_NORMALIZATION_MODE` | How the scale is set |
| --- | --- |
| `"image_zscore"` | Mean and spread of the whole image volume, as PyRadiomics does. The mask is not used. |
| `"image_percentile"` | The brighter half of the volume is clipped at `INTENSITY_PERCENTILES` and mapped onto a fixed range. The mask is not used, so the tumour keeps its brightness relative to the surrounding tissue. |
| `"tumor_zscore"` | Mean and spread of the voxels inside the tumour outline. Ties every pixel value to the outline. |

`USE_N4_BIAS_CORRECTION = True` removes the slow shading caused by the receive coils before anything else. It takes a few seconds per patient on first preparation and is then cached.

`FRONTEND_MODE = "on"` replaces the grey image, normally copied into all 3 network channels, with 3 fixed channels of your choice. Nothing in them is learned. Each channel is 1 of:

| Channel | Meaning |
| --- | --- |
| `"intensity"` | The normalised image, as in the default. |
| `"rank"` | The rank of each pixel among all voxels of the scan, which is unaffected by any change of the intensity scale. |
| `"energy:<size in mm>"` | Texture strength at that size, in all directions. |
| `"gradient:<size in mm>"` | Edge strength at that size. |
| `"window:<low>,<high>"` | For CT, Hounsfield units from low to high. |
| `"pre"`, `"early"`, `"late"`, `"washin"`, `"washout"` | Contrast phases and enhancement, for the NIfTI layout. |

Every size is given in millimetres and converted with the voxel spacing in use, so 1 setting describes the same physical structure at any resolution.

---

## 9. Backbones and global-local 3D views

```python
SHARED_BACKBONES = ['radimagenet_resnet10t', 'mars_3d']
MARS_USE_GLOBAL_LOCAL      = True
FMCIB_USE_GLOBAL_LOCAL     = True
GLOBAL_LOCAL_CROP_FRACTION = 0.5
```

List the backbones you want from the overview table above. Weights for the RadImageNet models must be downloaded by you. DINOv3, ConvNeXt, MARS and FMCIB weights are downloaded automatically on first use. `HF_TOKEN` is an optional Hugging Face access key and can normally stay empty.

**3D backbones.** MARS and FMCIB read a 3D block around the whole lesion, so the plane and slice settings do not affect them. Their main settings are the block size (`MARS_INPUT_SIZE`, `FMCIB_INPUT_SIZE`) and the extra tissue kept around the lesion (`MARS_CROP_MARGIN_MM`, `FMCIB_CROP_MARGIN_MM`).

**Global and local views.** With the global-local switches on, each patient is described by the whole lesion plus 7 smaller views centred on the lesion centre and its 6 outermost points. All 8 views pass through 1 shared network and the classifier receives the whole-lesion features together with the averaged local features. Runs with this option enabled write to their own results folder.

---

## 10. Classical machine learning (Section 2)

```python
RUN_CLASSIFIERS = ['NAIVE_BAYES', 'LOGREG', 'SVM']
```

Available classifiers are `NAIVE_BAYES`, `LOGREG`, `SVM`, `LDA`, `ADABOOST`, `RANDOM_FOREST`, `DECISION_TREE`, `XGBOOST`, `GAUSSIAN_PROCESS`, `AE`, `EXTRATREES`, `HIST_GBDT`, `SGD_LOGLOSS`, `LIGHTGBM`, `CATBOOST` and `BAGGING`.

Nested cross-validation runs on every feature table. An outer loop scores the model on held-out development patients while an inner loop tunes the settings on the remaining ones only, so no model is tuned on the patients used to score it. Every step that looks at the data, including rescaling, feature selection and class balancing, happens inside the folds.

| Step | Setting | Choices |
| --- | --- | --- |
| Rescaling | `SCALING_METHOD` | `zscore` or `minmax` |
| Class imbalance | `BALANCING_METHOD` | `class_weight` (default), `upsample`, `smote` or `none` |
| Dimension reduction | `PCA_VARIANTS`, `PCA_VARIANCE_GRID` | Number of components tried inside the folds |
| Feature selection by voting | `USE_FEATURE_SELECTOR_VOTING`, `RUN_FEATURE_SELECTOR_COMBINATIONS` | mRMR, Boruta, sparse logistic regression, elastic net and recursive elimination. A feature is kept when enough methods choose it. |
| Final feature ranking | `USE_POSTHOC_IMPORTANCE_FS` | Rank features with the best models, keep the top ones and retrain |

Cut-offs are set by Youden's J. Each outer fold uses the cut-off of its own training patients, and the test set is classified at the cut-off taken from the pooled development predictions. One fitted model per classifier is saved for later reuse.

---

## 11. Radiomics features

```python
RUN_RADIOMICS         = True
RADIOMICS_PARAMS_PATH = None          # or your own Params.yml
RADIOMICS_IMAGES      = ("early", "washin")
RADIOMICS_CROP_MARGIN_MM = 25.0
```

`RUN_RADIOMICS`, `RUN_LATE_FUSION` and `RUN_TOKEN_AGGREGATOR` are in the main settings block at the top, next to the other workflow switches. The detailed settings of each stay in their own cell.

PyRadiomics features are computed in 3D on the native image grid, once per patient, and saved to Google Drive immediately. A later run, with any backbone or workflow, extracts only new patients or changed settings. No labels are used at any point.

The built-in settings normalise the image, resample to 1 mm voxels, use a fixed bin width and compute every feature class on the original image and on Laplacian-of-Gaussian filtered versions. Set `RADIOMICS_PARAMS_PATH` to your own parameter file to replace them. `RADIOMICS_IMAGES` chooses the images the features are taken from, the analysed phase (`"early"`) and the enhancement (`"washin"`). Asking for `"washin"` makes the pre-contrast phase a required file, so a patient without it is left out of the whole run and listed with the reason in the pairing audit. Every workflow then analyses the same patients, which keeps the result tables comparable.

Section 2 then models a radiomics-only table and 1 combined CNN plus radiomics table per CNN table, through the same folds as everything else. A late-fusion step (`RUN_LATE_FUSION`) averages the standardised scores of the best CNN model and the best radiomics model and lists CNN, radiomics, early fusion and late fusion side by side.

---

## 12. Fine-tuning options (Section 3)

All Section 3 settings are in the block `FINE-TUNING SETTINGS (Section 3)` just below the main settings. Training runs in 2 phases, first the head alone and then the head together with part of the network. The backbone learning rate is chosen inside the development data and never assumed.

| Setting | What it does |
| --- | --- |
| `FINETUNE_TRAINABLE` | How much of each network is adjusted. `"auto"` (a small built-in part), a number of blocks such as `3`, a share such as `"50%"` or `"all"`. |
| `USE_BACKBONE_FINETUNE_PROTOCOLS` | Ready-made settings per backbone and cohort size, for example a cautious protocol below 600 patients and a longer one above 1000 patients. |
| `USE_LORA`, `LORA_BACKBONES` | LoRA adjusts a small number of extra weights instead of the original ones, which lowers the risk of overfitting a small dataset. |
| `USE_WISE_FT`, `WISE_FT_ALPHA` | Mixes the fine-tuned network with the network from before fine-tuning. Often more stable on data from another hospital, at no extra training cost. |
| `USE_MOVEMENT_CAP`, `FINETUNE_MAX_REL_RMS_MOVE` | Stops fine-tuning when the network moves too far from its pretrained weights, so pretraining is not lost. |
| `L2SP_LAMBDA`, `FINETUNE_LAYER_DECAY` | Hold the network near its pretrained weights, and let early layers move less than late ones. |
| `USE_EMA`, `USE_SWA` | Average the weights over training instead of using the last ones. Off by default. |
| `USE_TTA`, `USE_TTA_ROTATIONS` | Average the prediction over flipped and rotated copies of the image. Off by default. |
| `USE_MIXUP`, `LABEL_SMOOTHING`, `DROP_PATH_RATE` | Extra regularisation during training. |
| `USE_STAGED_UNFREEZE` | Release the network gradually instead of all at once. |
| `SLICE_POOLING` | How several planes are combined into 1 patient prediction, by averaging or by attention. |
| `FINETUNE_N_FOLDS`, `FINETUNE_N_REPEATS` | Cross-validation folds and repeats. |
| `USE_AMP`, `USE_CHANNELS_LAST`, `USE_GPU_RESIDENT_DATA` | Speed and memory settings that do not change the method. |

Runs with non-default fine-tuning settings write to their own results folders, so they never overwrite the default results.

`FROZEN_E2E_FINAL_MODEL` chooses how Section 1 scores the test patients, either `"refit"` (1 model retrained on all development patients) or `"fold_ensemble"` (the average of the cross-validation models).

---

## 13. Comparison runs

```python
RUN_SLICE_PLANE_COMPARISON = True
SLICE_PLANE_COMBINATIONS   = (("axial",), ("axial", "sagittal", "coronal"), ("coronal",))

RUN_PARAMETER_OPTIMIZATION = False
OPTIMIZATION_INPUT_SIZES   = (128, 224)
OPTIMIZATION_AUG_LEVELS    = ("FLIPS", "BASIC", "COMPLEX")
EVALUATE_HOLDOUT_FOR_ALL_COMPARISON_ARMS = True
```

`RUN_SLICE_PLANE_COMPARISON` tests several plane choices in 1 run on exactly the same patient splits. `RUN_PARAMETER_OPTIMIZATION` tests every combination of the listed image sizes in pixels and augmentation levels. Each tested combination is compared with a reference setting on the same patients and folds, and the comparison table reports the difference with its confidence interval.

With `EVALUATE_HOLDOUT_FOR_ALL_COMPARISON_ARMS = False`, the choices are compared on development data only and the test patients stay completely unseen.

---

## 14. Additional segmentations and mask agreement

```python
RUN_AUTOSEG_NNINTERACTIVE   = True
RUN_AUTOSEG_SAM2            = False
ANALYZE_NNINTERACTIVE_MASKS = True
ANALYZE_SAM2_MASKS          = False
NNINTERACTIVE_ANALYSIS_PROMPT_TYPE = "lasso"
```

**Purpose.** These functions answer 2 questions. First, how closely does an automatic segmentation agree with the manual one, which shows how much the tumour outline varies between a human and an automatic method. Second, do the classification results change when automatic masks are used instead of manual ones.

The code creates extra tumour masks with **nnINTERACTIVE** and **CAS2 / SAM 2.1** and compares each one with the original mask using the **Dice Similarity Coefficient (DSC)**. A DSC of `1` means the masks match exactly and `0` means they do not overlap. DSC measures agreement with the original mask and does not prove that either mask is correct.

The existing masks provide the guidance, called **prompts**, so these are guided resegmentations rather than independent ones.

- **nnINTERACTIVE** works on the whole 3D image. It receives 1 prompt on the slice where the tumour is largest and draws the full 3D mask itself. `NNINTERACTIVE_ANALYSIS_PROMPT_TYPE` chooses the prompt, either `"lasso"` (a filled outline), `"box"` (a tight rectangle), `"scribble"` (a short stroke inside the tumour) or `"point"` (1 point well inside the tumour). Only the chosen type is created and analysed in a run.
- **CAS2 / SAM 2.1** works slice by slice in 2D and receives 1 box around the tumour on every slice containing tumour. Part of its agreement is therefore built in. The column `dice_box_only_control` shows how much, as the DSC of a simple oval drawn inside the same boxes without looking at the image.

Creating masks and analysing them have separate switches.

| Setting | What it enables |
| --- | --- |
| `RUN_AUTOSEG_NNINTERACTIVE` | Creates or loads nnINTERACTIVE masks and compares them with the original masks. |
| `RUN_AUTOSEG_SAM2` | Creates or loads CAS2 / SAM 2.1 masks and compares them with the original masks. |
| `ANALYZE_NNINTERACTIVE_MASKS` | Repeats the selected workflows using the nnINTERACTIVE masks. |
| `ANALYZE_SAM2_MASKS` | Repeats the selected workflows using the CAS2 / SAM 2.1 masks. |

To create and analyse a mask type, set both of its switches to `True`. To measure agreement only, switch on `RUN_AUTOSEG` and switch off `ANALYZE`. To use masks saved in an earlier run, switch off `RUN_AUTOSEG` and switch on `ANALYZE`. With both `RUN_AUTOSEG` switches `False`, the segmentation cell is skipped and its extra packages are not installed.

**Saved masks.** New masks are saved as 1 NRRD file per patient in Google Drive, with a separate folder per mask type.

| Setting | Mask type |
| --- | --- |
| `CAS2_ROOT` | CAS2 / SAM 2.1 |
| `LASSO_ROOT`, `BOX_ROOT`, `SCRIBBLE_ROOT`, `POINT_ROOT` | nnINTERACTIVE with that prompt |

Before creating a mask, the code checks whether it already exists. Existing masks are reused and only missing masks are created, which saves processing time in later runs. To create a patient's mask again, delete or rename that patient's saved file first.

**Results.** Agreement tables and overlay pictures go to `OUT_ROOT/AUTO_SEGMENTATION_CHECK`.

- `auto_segmentation_dice_per_patient.csv`: 1 row per patient and method, with DSC, DSC of the largest lesion, tumour volumes and whether the same slice was chosen for analysis.
- `auto_segmentation_dice_summary.csv`: median, quartiles, mean and lowest DSC per method, and how many patients fell below `AUTOSEG_LOW_DICE` (default 0.70).
- Overlay figures showing the manual and automatic outlines for a few patients.

Classification results with the extra masks go to `OUT_ROOT/SEGMENTATION_ANALYSES/`, and the main overview table then marks every row in its `segmentation_method` column.

**Model weights.** Set `NNINTERACTIVE_WEIGHTS_PATH` and `SAM2_WEIGHTS_PATH` to the folders holding the downloaded weights, or set `AUTOSEG_DOWNLOAD_MISSING_WEIGHTS = True` to let the notebook download them once.

- nnINTERACTIVE v1.0, about 410 MB, non-commercial licence CC BY-NC-SA 4.0: https://huggingface.co/MIC-DKFZ/nnInteractive/tree/main/nnInteractive_v1.0
- SAM 2.1 Hiera Base+, about 325 MB, Apache-2.0 licence: https://huggingface.co/facebook/sam2.1-hiera-base-plus/tree/main

**Test on a few patients first.** `AUTOSEG_PILOT_N_PATIENTS` runs the segmentation on that many development patients only. Set it back to `None` for a full analysis.

---

## 15. Contour perturbation test (Section 7)

```python
RUN_MASK_PERTURBATION_RETEST = True
PERTURB_RETEST_ARMS   = ("finetune_cnn",)
PERTURB_RETEST_LEVELS = (
    ("dilate", 1.0), ("dilate", 2.0), ("dilate", 3.0),
    ("erode", 1.0), ("erode", 2.0), ("erode", 3.0),
    ("shift", 2.0), ("shift", 5.0),
    ("scale", 0.1), ("scale", -0.1),
    ("deform", 1.0), ("deform", 2.0), ("deform", 3.0),
)
```

**Purpose.** The tumour mask decides which slice is analysed and where the image is cropped, and with `"tumor_zscore"` it also sets the intensity scale. Two readers rarely draw exactly the same outline. This test answers the question **"how much would the result change if the tumour outline had been drawn slightly differently?"** It also shows how accurate an automatic segmentation would have to be to replace manual outlining.

**How it works.** No model is retrained. The models saved by an earlier full run are reloaded and kept unchanged. Only the tumour mask of each **test** patient is changed and the test patients are scored again, so any change in the result comes from the outline alone.

| Change | Meaning | Size given in |
| --- | --- | --- |
| `"dilate"` | Enlarges the outline on all sides. | Millimetres |
| `"erode"` | Shrinks the outline on all sides. | Millimetres |
| `"shift"` | Moves the whole outline in a random direction. | Millimetres |
| `"scale"` | Enlarges or shrinks the outline by a fraction, so `0.2` is 20% larger and `-0.1` is 10% smaller. | Fraction |
| `"deform"` | Moves the outline in and out by a smooth random amount, which imitates 2 readers disagreeing locally. | Millimetres |

| Setting | What it does |
| --- | --- |
| `PERTURB_RETEST_ARMS` | Which saved models to test, `"finetune_cnn"` (Section 3) or `"frozen_ml"` (Section 2) or both. Keep the comma for 1 entry. |
| `PERTURB_RETEST_CNN_MODEL_PATH`, `PERTURB_RETEST_ML_MODEL_PATH` | The saved model files to test. `None` searches the results folder and stops with a list if it finds more than 1 match. |
| `PERTURB_RETEST_BACKBONE` | The backbone of the saved model. `None` reads it from the file itself. |
| `PERTURB_RETEST_TEST_SET` | Which test dataset to use, named after its folder. |
| `PERTURB_RETEST_SHIFT_REPEATS` | Random draws per patient for `"shift"` and `"deform"` (default 5 draws). The other changes are not random and are done once. |
| `PERTURB_RETEST_SEED` | Fixes the random draws, so a rerun gives exactly the same changed masks. |
| `PERTURB_RETEST_SAVED_MASK_METHODS` | Adds the saved nnINTERACTIVE and SAM 2 masks as real-world contours. |
| `PERTURB_RETEST_MASK_DIR` | Where the changed masks are saved and reused. `None` uses `OUT_ROOT/MASK_PERTURBATION`. |

**Before you use it.** A full run must have finished first, because this test reloads its saved models. You can then set `RUN_CNN_FINETUNE_E2E = False` and rerun to repeat only the perturbation test. Nothing is retrained and no GPU is needed.

**Results** in `OUT_ROOT/MASK_PERTURBATION_RETEST`:

- `perturbation_summary.csv`, 1 row per change, with
  - `dsc_median`, how well the changed outline agrees with the original one. Published reader-to-reader DSC values for meningioma are about 0.85 to 0.93, so rows in that range represent realistic disagreement between human readers.
  - `same_central_slice_pct`, how often the same slice is still chosen for analysis.
  - `auc_mean` and `auc_sd`, the test-set AUC and its spread over the random draws.
  - `auc_delta_vs_original` with `delta_ci_low` and `delta_ci_high`, the AUC change against the original outline with its confidence interval. An interval containing 0 means the change did not measurably affect the result.
- `perturbation_per_pass.csv`, 1 row per model, change and random draw.
- `auc_against_dsc.png`, a figure of AUC against DSC. The DSC at which the AUC starts to fall is the quality an automatic segmentation would have to reach.

---

## 16. Calibration, precision-recall and decision curves (Section 8)

```python
RUN_CALIBRATION_AND_DECISION_CURVES = True
```

Nothing is trained and no earlier result changes. The AUC only says how well a model ranks patients, so this section reads the saved per-patient predictions and adds:

- **PR-AUC**, to be compared with the share of positive patients, which is the value of a model without information. More informative than AUC when the positive class is rare.
- **Calibration intercept**, where 0 is ideal, showing whether predicted risks are too high or too low on average.
- **Calibration slope**, where 1 is ideal. Below 1 the predictions are too extreme, the usual sign of overfitting.
- **Decision curve**, the net benefit of acting on the model at each risk threshold, against treating every patient and treating none.

Both workflows shift predicted probabilities towards the smaller class, through class weighting or balancing. The prior-corrected probability has that shift removed and is the scale on which calibration and net benefit should be judged. Outputs go to `CNN_ANALYSIS_ROOT/CALIBRATION_AND_DECISION_CURVES`.

---

## 17. Saliency check (Section 9)

```python
RUN_SALIENCY_CHECK  = True
SALIENCY_N_PATIENTS = 12
```

A grey patch is moved over each analysed slice and the change in the model output is recorded at every position. Regions whose covering changes the output most are the regions the model relies on. The method needs no access to the inside of the network, so it works for every 2D backbone, including the Vision Transformers.

For each patient the cell reports the share of this sensitivity that lies inside the tumour mask, the share of the crop the tumour covers and their ratio. A ratio above 1 means the model responds to the tumour more than to an equally large random part of the crop. Ratios near or below 1 in most patients mean the prediction is driven by the surroundings and should be examined before the model is trusted. Outputs go to `CNN_ANALYSIS_ROOT/SALIENCY_CHECK`.

---

## 18. Results at the end of a run

- **`combined_pipeline_overview.csv`** in `COMBINED_OVERVIEW_DIR` is the main result table. It lists every model of every enabled workflow and mask type, with cross-validation results on the development patients and results on the test patients. The `workflow` column reads `frozen`, `efficient_ML` or `finetune`, and `segmentation_method` reads `original`, `sam2_box_per_slice` or `nninteractive_<prompt type>`. With 2 test datasets, 1 extra file is written per test dataset.
- **Results dashboard.** The last cell repeats every result table and saves `all_registered_result_tables_long.csv` (all tables combined), an index of those tables and an index of every CSV file written during the run.
- **Audit files** in `OUT_ROOT/SHARED_INPUTS_AUDIT` list excluded patients, with the reason, together with lesion counts, intensity clipping and image-mask geometry checks.
- AUC values are reported with confidence intervals. `TOP_RISK_PERCENT` also reports results for the patients with the highest predicted risk. `Brier_score` is reported as predicted and as prior-corrected.

---

## Repository contents

- **Code_1, DICOM or MRB to NRRD conversion.** Converts raw DICOM or MRB data into paired NRRD image and mask files.
- **Code_2, analysis pipeline.** The main notebook. It reads either the NRRD image and mask pairs created by Code_1 (Option A) or NIfTI image and mask folders (Option B).

## CNN weights

Download the pretrained RadImageNet weights for the models you use and place the files in the folder set by `WEIGHTS_BASE`.

```python
WEIGHTS_BASE = "/content/drive/MyDrive/DATA/weights"
```

- DenseNet121: https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/DenseNet121.pt?download=true
- ResNet50: https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt?download=true
- ResNet18: https://huggingface.co/convergedmachine/RadImagenet/resolve/main/resnet18.pth?download=true
- ResNet10t: https://huggingface.co/convergedmachine/RadImagenet/resolve/main/resnet10t.pth?download=true
- InceptionV3: https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/InceptionV3.pt?download=true

These weights are downloaded automatically and need no manual step.

- ImageNet models, through the standard Python model libraries.
- DINOv3 and ConvNeXt, through Hugging Face and timm: https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m
- MARS, downloaded once to `MARS_WEIGHTS_PATH`: https://github.com/zqiuak/MARS
- FMCIB, downloaded once to `FMCIB_WEIGHTS_PATH`, about 0.7 GB: https://zenodo.org/records/10528450
- nnINTERACTIVE and SAM 2.1, see setting 14.

## Further instructions

Detailed instructions for every setting are given in the first notebook cell (Section 0, cell 1) and in the comments within the code itself.

## How to cite

If you use this code in a published study, please cite it. You can use the "Cite this repository" button on the right side of the GitHub page or the following reference.

(submitted for publication)
[name/authors]. (2026). [Repository Name]. GitHub. [GitHub URL]
