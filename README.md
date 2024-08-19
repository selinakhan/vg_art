# Context-Infused Visual Grounding for Art

This repository contains the code used for the experiments in "Context-Infused Visual Grounding for Art" from the ECCV 2024 VISART workshop.  

This research adopts multiple code bases and should be structured as follows: 
```
vg_art
│   README.md
│   inference.py    
│	metrics.py
│	requirements.txt
│
└─── util
   │   misc.py
   │   visualize.ipynb
│
└─── data_refinement
   │   postprocess_refinement.py
   │   refinement_util.py
│
└─── Open-GroundingDino (external)
   │  ...
│
└─── GroundingDINO (external)
   │  ...
```


## Data Preparation
All datasets used to train CIGAr should be converted into ODVG format as described [here](https://github.com/longzw1997/Open-GroundingDino/blob/main/data_format.md). For reproducibility, the IconArt, ArtDL and DEArt (Captions) ground-truth and pseudo-ground-truth annotations as used in this research have been converted to the required formats and can be found [here](https://drive.google.com/drive/folders/1MnVRm-j0ImfGdLt1miqNUyPxGLiPlo7o?usp=sharing). The same applies to the [Ukiyo-eVG dataset](https://drive.google.com/drive/folders/1cLhue-10GYtjybQ5CMPRyFcI_LjgArcE?usp=drive_link).  Images should be retrieved from the original sources. DEArt and DEArt Captions both use the same image set (although DEArt Captions adopt a subset). All datasets with ground truth object detection annotations include a `label_map.json` and the (test/val) ground truth annotations in COCO format as required for the evaluation used in [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino).

## Code Preparation

For this research, the code from [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino) has been adapted to include artwork descriptions during training and to evaluate on grounding data. The adapted code used can be found [here](https://github.com/selinakhan/Open-GroundingDino), and should be cloned into this repository. Additionally, the original [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repository should be cloned. Future efforts might make this step redundant, but as the Open-GroundingDino code currently depends on it, this is required for now.  

In a single environment, follow the setup instructions from Open-GroundingDino to build the code, and do the same for the original GroundingDINO repository. Additionally install the required packages from this repository found in `requirements.txt`. Our experiments are conducted using `torch v. 2.0.1` with `cuda 11.7`, but other versions may also work.

## Models

Model checkpoints for each configuration which achieved the best performance in this research are listed below. CIGAr models include the fine-tuned BERT backbone.

| Dataset | Model | mAP@0.5|
|--|--|--|
|Ukiyo-eVG  | [CIGAr](https://drive.google.com/drive/folders/1ulg5BVqj1vVhMYXaqwndOAA78WlJGPVA?usp=sharing) | 26.8|
| DEArt Captions | [CIGAr](https://drive.google.com/drive/folders/1hkpQzDKNrslKLJ_FHm63FnY-9wEqRaZj?usp=sharing) | 30.3|
| DEArt | [Fine-tuned GroundingDINO](https://drive.google.com/drive/folders/1cLs_Tmq7tk6biZMs8nufUQSMNz6A6Xa6?usp=sharing) * | 25.3|
| IconArt | [Fine-tuned GroundingDINO](https://drive.google.com/drive/folders/1qRiGG8PxuCvxXBO-zKYvLnDL0OA59bYc?usp=sharing) | 35.2 |
| ArtDL | [Fine-tuned GroundingDINO](https://drive.google.com/drive/folders/1qRiGG8PxuCvxXBO-zKYvLnDL0OA59bYc?usp=sharing) | 51.1|

\*Not state-of-the-art performance.


## Training
For instructions to train CIGAr, please refer to the adapted Open-GroundingDino repo as found [here](https://github.com/selinakhan/Open-GroundingDino/).

## Evaluation
For model inference, run the following command on any dataset in ODVG format.

`` python inference.py
	--data_path [path to ODVG dataset]
	--cap_style [input prompt: full/entities/relations] 
	--dataset_type [OD/VG] 
	--label_mapping [path to label map required for OD datasets] 
	--img_path [path to image folder]
	--model_checkpoint_path [path to .pth model checkpoint]
	--model_config_path [path to .py config file]
	--text_threshold [float]
	--box_threshold [float]
	--output_path [file path to save inference results in .jsonl]``

To perform evaluation on datasets in ODVG format, run

``python metrics.py
--gt_data [path to ODVG ground truth]
--pred_data [path to ODVG predictions]
--phrase_matching [exact/fuzzy]
--dataset_type [od/vg]``

## Data Refinement

[TODO]
