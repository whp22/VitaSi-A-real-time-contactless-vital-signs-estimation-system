# VitaSi: A real-time contactless vital signs estimation system
This is the official implementation of paper:["VitaSi: A real-time contactless vital signs estimation system"](https://www.sciencedirect.com/science/article/abs/pii/S0045790621003591)

# System Overview
The inputs are RGB frames from an exercise video. The whole system is mainly composed of 4 parts: MSPN 2D human pose estimation model, joint location calculation, heatmap processing and the multitask model for exercise recognition & counting.
![](/images/system.JPG)

# Multitask Model
![](/images/model.JPG)
# Requirements
- Tensorflow 1.16
- Python 3.6
# Dataset Preparation
Rep-Penn Dataset is not provided here. If you want to create the dataset in the same way, please refer to our paper.

The optional method is generating a heatmap for one-cycle exercise videos, and duplicate&concatenate heatmaps using similar methods introduced in the paper.
# Running the code
## Training
Train from scratch. Please change the keywords ('action' or 'counting') to train corresponding branch. 
```
python3 train_multitask.py
```

## Testing
```
python3 eval_multitask.py
```

# Citation
If you use this code, please cite the following:
```
@article{WANG2021107392,
title = {VitaSi: A real-time contactless vital signs estimation system},
journal = {Computers and Electrical Engineering},
volume = {95},
pages = {107392},
year = {2021},
issn = {0045-7906},
doi = {https://doi.org/10.1016/j.compeleceng.2021.107392},
url = {https://www.sciencedirect.com/science/article/pii/S0045790621003591},
author = {Haopeng Wang and Yufan Zhou and Abdulmotaleb El Saddik},
}
```

