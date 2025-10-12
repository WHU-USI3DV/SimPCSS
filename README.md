# SimPCSS

<h2> 
<a href="https://github.com/WHU-USI3DV/SimPCSS" target="_blank">Simulated Point Clouds Explicitly Guided Semantic Segmentation</a>
</h2>

This is the PyTorch implementation about point cloud semantic segmentation of the following publication:

> **Simulated Point Clouds Explicitly Guided Semantic Segmentation**<br/>
> [Zhe Chen](https://chenzhe-code.github.io/), [Jiahao Zhou](https://ddakun.github.io/), [Chen Long](https://chenlongwhu.github.io/), [Peiling Tong](https://3s.whu.edu.cn/info/1028/1961.htm), [Pangyin Li](https://3s.whu.edu.cn/info/1028/2062.htm), [Fuxun Liang](https://liangfxwhu.github.io/), [Zhen Dong](https://dongzhenwhu.github.io/index.html) <br/>
> [**Paper**](https://github.com/WHU-USI3DV/SimPCSS)  *PE&RS 2025*<br/>


## 🔭 Introduction
<p align="center">
<strong>Simulated Point Clouds Explicitly Guided Semantic Segmentation</strong>
</p>
<div align=center>
<img src="media/teaser.png" alt="Network" style="zoom:30%" align='middle'>
</div>

<p align="justify">
<strong>Abstract:</strong>
Point cloud semantic segmentation (PCSS) is crucial for smart city management but remains a challenging task due to the irregular and sparse nature of the data. While recent advancements in PCSS focus on improving network architectures, less attention has been given to the data aspect. In image analysis, synthetic data has proven useful, but generating point clouds that match real-world distributions remains difficult. In contrast, it is accessible to obtain unlimited high-density, noise-free point clouds through simulators. To enhance PCSS from the data aspect, we propose Simulated Point Clouds Explicitly Guided Semantic Segmentation (SimPCSS), a plug-and-play supervised learning scheme. Specially, we generate labeled point clouds in various scenarios using an autonomous driving simulator and train a segmentation model. Then, multi-scale features with high confidence are then extracted to construct prior guidance through the Confidence Update Strategy (CUS). We further introduce an Imitation Learning Strategy (ILS), which injects the above prior guidance into the segmentation process of low-quality point clouds, improving performance. The proposed method is model-agnostic, requiring only minor adjustments to existing network architectures. Experiments conducted on both synthetic and real-world datasets with various models (MinkUnet & PTv3) demonstrate that SimPCSS effectively leverages high-quality point clouds to improve the segmentation of low-quality point clouds.
</p>

## 🆕 News
- 2025-10-10:  Accepted by PE&RS! 🎉🎉🎉
- 2023-10-5: Code is aviliable! 🎉


### 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- CUDA 11.6
- Python 3.8.0
- Pytorch 1.13.1
- GeForce RTX 4090.

### 🔧 Installation
You can create an environment directy using the provided ```environment.yaml```
```
conda env create -f environment.yaml
conda activate pcda
```

### 💾 Dataset 
Our method has been experimented in both the benchmark and practical applications.
>- **ISPRS 2D cross-domain semantic segmentation benchmark**  
&ensp;&ensp;&ensp;&ensp;Provided by [Te Shi](https://github.com/te-shi/MUCSS?tab=readme-ov-file), the ISRPS image dataset for cross-domain semantic segmentation can be downloaded via [Google Drive](https://drive.google.com/file/d/1amV--tjtjBMUscUVBqXxXws_vBCo-QdV/view) or [BaiduDisk](https://pan.baidu.com/share/init?surl=Ob12TozQ2Xjcm3rcv7LuRA) (Acess Code: vaam).
>- **Pratical applications: SpaceNet-Shanghai to GES-Wuhan**  
&ensp;&ensp;&ensp;&ensp;The SpaceNet-Shanghai dataset and GES image dataset ( Wuchang District, Wuhan ) can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1l5ARaev5hO95LG1e3e7top-Gda3y8BGb?usp=sharing) or [BaiduDist](https://pan.baidu.com/s/1qmGYUhlGQ9mJGvgez-bbwQ ) (Acess Code: 9527)

Once the datasets are downloaded and decompressed, change the folder path of the dataset according to the actual path in file *```UDA-Seg/core/datasets/dataset_path_catalog.py```* (line 33-36) for training and testing purposes.

### 🔦 Train
We provide the training script for source domain training and domain adaptation training. 
```
cd UDA-Seg
bash train_with_sd.sh
```
Specially, supervised training on labeled source domain data is needed to initialize the network parameters firstly.
```
# Set the num of GPUs, for example, 2 GPUs
export NGPUS=2
# train on source data
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/rs_deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/
```
Then, we conduct the unlabeled target domain data and labeled source domain data for adversarial training to carry out domain adaptation.
```
# train with FGDAL-MSF-DNT
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_SEmsf_dnt_adv_BD+FD_FreezeBackbone.py -cfg configs/rs_deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_try/model_iter030000.pth
```
Note that our framework does not use **self distill**. However, you can slightly modify the code (network) in *```train_self_distill.py```* to conduct self distill and further improve the performance.

```
# generate pseudo labels for self distillation
python test.py -cfg configs/rs_deeplabv2_r101_tgt_self_distill.yaml --saveres resume results/adv_test/model_iter080000.pth OUTPUT_DIR datasets/rs/soft_labels DATASETS.TEST rs_train
# train with self distillation
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_self_distill.py -cfg configs/rs_r101_tgt_self_distill.yaml OUTPUT_DIR results/sd_test
```
### ✏️ Test

```
# Test the performance of a specific ckpt
python test_SEmsf_dnt.py -cfg configs/configs/rs_deeplabv2_r101_adv.yaml --saveres resume results/adv_test/model_iter080000.pth OUTPUT_DIR datasets/rs/pred
# Test the performance of all ckpt in a folder
python test_SEmsf_dnt.py -cfg configs/configs/rs_deeplabv2_r101_adv.yaml --saveres resume results/adv_test OUTPUT_DIR datasets/rs/pred
```
Additionaly, for predicting a RS image with large range, we provide a predition code based on sliding window.
```
python predcit_dnt_SEmsf_large_image.py -cfg configs/configs/rs_deeplabv2_r101_adv.yaml --img_path test_img/Wuchang_GES_Image.tif
```

## 🚅 Building height deriving from global DSM
### 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- Python 3.8.0

### 🔧 Installation
Fewer packages are required for building height extraction:
```
conda create -n BHD python==3.8
conda activate BHD
pip install matplotlib tqdm gdal argparse opencv-python
```

### 💾 Dataset 
The DSM with global coverage of 30m resolution is accessible in [Japan Aerospace Exploration Agency  Earth Observation Research Center](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/index.htm).


### 🔦 Usage
To derive the building height from global DSM, you can use the following commands:
```
python BH_ExtractionFromAW3D30.py --input_path AW3D30_WH.tif --output_path BH_WH.tif
```

## 💡 Citation
If you find this repo helpful, please give us a 😍 star 😍.
Please consider citing our works if this program benefits your project.
```
@article{CHEN2024122720,
  title = {City-Scale Solar {{PV}} Potential Estimation on {{3D}} Buildings Using Multi-Source {{RS}} Data: {{A}} Case Study in {{Wuhan}}, {{China}}},
  author = {Chen, Zhe and Yang, Bisheng and Zhu, Rui and Dong, Zhen},
  year = {2024},
  journal = {Applied Energy},
  volume = {359},
  pages = {122720},
  issn = {0306-2619},
  doi = {10.1016/j.apenergy.2024.122720}
}

@article{chenJointAlignmentDistribution2022,
  title = {Joint Alignment of the Distribution in Input and Feature Space for Cross-Domain Aerial Image Semantic Segmentation},
  author = {Chen, Zhe and Yang, Bisheng and Ma, Ailong and Peng, Mingjun and Li, Haiting and Chen, Tao and Chen, Chi and Dong, Zhen},
  year = {2022},
  month = dec,
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  volume = {115},
  pages = {103107},
  issn = {1569-8432},
  doi = {10.1016/j.jag.2022.103107},
  urldate = {2022-12-02},
  langid = {english}
}

```

## 🔗 Related Projects
We sincerely thank the excellent project:
- [FADA](https://github.com/JDAI-CV/FADA) for UDA semantic segmentation;
- [FreeReg](https://github.com/WHU-USI3DV/FreeReg) for readme template.
