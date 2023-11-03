# VitaSi: A real-time contactless vital signs estimation system
This is the official implementation of paper:["VitaSi: A real-time contactless vital signs estimation system"](https://www.sciencedirect.com/science/article/abs/pii/S0045790621003591)

# System Overview
The non-contact monitoring of vital signs, especially the Heart Rate (HR) and Breathing Rate (BR), using facial video is becoming increasingly important. Although, researchers have made considerable progress in the past few years, there are still some limitations to the technology, such as the lack of challenging datasets, the time consuming nature of the estimation process, and non-portability of the system. In this paper, we proposed a new framework for estimating HRs and BRs by combining a Convolutional Neural Network (CNN) with the Phase-based Video Motion Processing (PVMP) algorithm. The experimental results show that our approach achieves better performance. Meanwhile, we introduce a new challenging dataset with fewer constraints, such as large movements, facial expressions and light interference. In addition, we developed a new Android application, which works in real time and offline, based on a CNN for HR and BR estimations.


# Requirements
- Tensorflow 1.16
- Python 3.6
- OpenCV


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

