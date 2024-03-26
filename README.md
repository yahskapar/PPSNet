# Leveraging Near-Field Lighting for Monocular Depth Estimation from Endoscopy Videos

<p align="center">
:fire: Please remember to :star: this repo if you find it useful and <a href="https://github.com/Roni-Lab/PPSNet#scroll-citation">cite</a> our work if you end up using it in your work! :fire:
</p>
<p align="center">
:fire: If you have any questions or concerns, please create an issue :memo:! :fire:
</p>

<p align="center">
<a href="https://arxiv.org/">Pre-print</a> | <a href="https://ppsnet.github.io/">Project Website</a>
</p>

## :book: Abstract

Monocular depth estimation in endoscopy videos can enable assistive and robotic surgery to obtain better coverage of the organ and detection of various health issues. Despite promising progress on mainstream, natural image depth estimation, techniques perform poorly on endoscopy images due to a lack of strong geometric features and challenging illumination effects. In this paper, we utilize the photometric cues, i.e., the light emitted from an endoscope and reflected by the surface, to improve monocular depth estimation. We first create two novel loss functions with supervised and self-supervised variants that utilize a per-pixel shading representation. We then propose a novel depth refinement network (PPSNet) that leverages the same per-pixel shading representation. Finally, we introduce teacher-student transfer learning to produce better depth maps from both synthetic data with supervision and clinical data with self-supervision. We achieve state-of-the-art results on the C3VD dataset while estimating high-quality depth maps from clinical data. Our code, pre-trained models, and supplementary materials can be found on our project page: https://ppsnet.github.io/.

## :wrench: Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate ppsnet` 

STEP3: `pip3 install -r requirements.txt`

STEP 4: Install PyTorch using the below command,

```
pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```
The exact versioning may vary depending on your computing environment and what GPUs you have access to. Note this [good article](https://webcache.googleusercontent.com/search?q=cache:https://towardsdatascience.com/managing-multiple-cuda-versions-on-a-single-machine-a-comprehensive-guide-97db1b22acdc&sca_esv=86c40678baae855f&sca_upv=1&strip=1&vwsrc=0) for maintaining multiple system-level versions of CUDA.

STEP 5: Download the [C3VD dataset](https://durrlab.github.io/C3VD/). Our preprocessing steps for the dataset involve performing calibration and undistorting the images (a script for which will be released in the near future). We've provided a validation portion of the dataset in a Google Drive for reference and ease-of-use with this repo's evaluation code. You can download that portion of the dataset [here (~29GB)](https://drive.google.com/drive/folders/1QfacGUjaD1-ByC1XvukUzu84HGdwKXhF?usp=sharing). Note the [original licensing terms](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1) of the C3VD data. 

STEP 5: Download the appropriate [pre-trained models](https://drive.google.com/drive/folders/17778hK9_Zk9lSrDr5EPQnmLEUwLnpblB?usp=sharing) and place them in a newly created folder called `checkpoints/`.

## :computer: Usage

You can evaluate our backbone model using the `test_backbone.py` script:
```
python3 test_backbone.py --data_dir /your/path/to/data/dir --log_dir ./your_path_to_log_dir --ckpt ./your_path_to_checkpoint
```
Similarly, our teacher model and our student model can be evaluated using the `test_ppsnet.py` script:
```
python3 test_ppsnet.py --data_dir /your/path/to/data/dir --log_dir ./your_path_to_log_dir --ckpt ./your_path_to_checkpoint
```
Both scripts will generate various folders in the specified `log_dir` containing input images, ground truth and estimate depths, and percent depth error maps. Please keep an eye on this repo for future updates, including a full release of the training code, baselines included in the paper, mesh generation and visualization code, and more.

## :scroll: Acknowledgments
Thanks to the authors of [Depth Anything](https://github.com/LiheYoung/Depth-Anything) and [NFPS](https://github.com/dlichy/FastNFPSCode) for their wonderful repos with open-source code!

## :scroll: Citation
If you find our [paper](https://arxiv.org/) or this toolbox useful for your research, please cite our work.

```
Coming soon!
```
