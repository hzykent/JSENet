# JSENet: Joint Semantic Segmentation and Edge Detection Network for 3D Point Clouds
## Introduction
Implementation of ECCV2020 paper - JSENet: Joint Semantic Segmentation and Edge Detection Network for 3D Point Clouds ([arXiv](https://arxiv.org/abs/2007.06888)). If you find our work useful in your research, please consider citing:

```
@inproceedings{hu2020jsenet,
  title={JSENet: Joint Semantic Segmentation and Edge Detection Network for 3D Point Clouds},
  author={Hu, Zeyu and Zhen, Mingmin and Bai, Xuyang and Fu, Hongbo and Tai, Chiew-lan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={222--239},
  year={2020}
}
```

## Installation

This repository is modified from [KPConv](https://github.com/HuguesTHOMAS/KPConv/), please find the step-by-step installation guide in [INSTALL.md](https://github.com/HuguesTHOMAS/KPConv/blob/master/INSTALL.md).

## Experiments

### Data

#### S3DIS

S3DIS dataset can be downloaded <a href="https://goo.gl/forms/4SoGp4KtH1jfRqEj2">here (4.8 GB)</a>. Download the file named `Stanford3dDataset_v1.2.zip`, uncompress the folder and move it to `Data/S3DIS`. 

We provide processed demo dataset for experiments on S3DIS fold-5. The demo dataset can be downloaded <a href="https://drive.google.com/file/d/1Zi8rdgFDWGtlHvaJ9icr6zdi0L4UV02X/view?usp=sharing">here (903 MB)</a>. Uncompress the folder and move it to `Data/S3DIS`.

#### Scannet

Scannet dataset can be find <a href="http://www.scan-net.org/">here</a>. Follow the instructions and move downloaded files to `Data/Scannet`.

### Training
For S3DIS dataset:
    
    python training_S3DIS.py
    
    
For Scannet dataset:
    
    python training_Scannet.py

If you are not using the processed demo dataset, the first run will take some time to process the raw data. The process can be easily accelerated using parallel computing methods like Pthreads.

### Testing

In `test_model.py`, you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script :

For semantic segmentation task:

    python test_model.py --task SS

For semantic edge detection task:

    python test_model.py --task SED


### Pretrained models

Pretrained models can be downloaded <a href="https://drive.google.com/file/d/14YJxEJ8Rg6_jjE8l5iYFF9wJTK2XxCc5/view?usp=sharing">here (119 MB)</a>. Uncompress the files and move it to `results/`.


## Acknowledgment

Our code is modified from [KPConv](https://github.com/HuguesTHOMAS/KPConv/).

## License
Our code is released under MIT License (see LICENSE file for details).
