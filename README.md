# configurable-leakage-controlled-multi-workflow-medical-image-classification-suite
A highly configurable Google Colab notebook comparing three leakage-controlled DL and ML workflows for medical image classification. Featuring a single-notebook design with clear top-level settings, it is easy to use for non-coding researchers yet fully customizable for experienced developers.

Repository Contents:
This repository consists of two primary components:

Code_1 (DICOM to NRRD Conversion): A script that converts raw DICOM data into paired NRRD image and mask files.

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
