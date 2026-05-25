# TempoCL: Temporal Semi-supervised Contrastive Learning under Open-set and Closed-set Noisy Labels

This is the official code for our paper "TempoCL: Temporal Semi-supervised Contrastive Learning under Open-set and Closed-set Noisy Labels".

## Abstract

Deep neural networks degrade significantly when trained on label-noisy data. In practice, label noise presents itself as either closed-set, where samples are misassigned to an existing class, or open-set, where samples do not belong to any existing class. Despite significant progress in existing research, two critical problems remain unaddressed: (1) Existing methods for addressing open-set and closed-set label noise typically rely on predefined thresholds. However, as target data distributions diverge from the training set, these static thresholds lose effectiveness, thereby hindering the ability of the model to accurately differentiate between open-set and closed-set noise. (2) Existing multi-view contrastive learning for label noise typically uses data augmentation for diverse views. However, time series data exhibits distinct scale-dependent patterns: fine scales capture microscopic features, while coarse scales reveal macroscopic information. Data augmentation rarely captures these complementary details effectively, thus limiting performance. To address these challenges, we introduce TempoCL, a temporal semi-supervised contrastive learning network. 
The core process of TempoCL unfolds as follows: (1) Two-stage Adaptive Sample Selection (TASS) leverages training loss in the first stage to separate clean samples from noisy ones. Subsequently, the second stage discriminates between closed-set and open-set noise by exploiting the fact that open-set instances fall outside the predefined categories. TempoCL applies supervised and semi-supervised learning to the clean and close-set samples, respectively.  (2) Subsequently, Multi-view Temporal Contrastive Learning (MvTCL) is applied to the clean samples. MvTCL ensures same-scale consistency among multi-view representations while enhancing different-scale complementarity, thereby bolstering robustness against label noise by effectively integrating microscopic and macroscopic information.
Extensive experiments on diverse standard time series datasets confirm TempoCL significantly outperforms current state-of-the-art methods.

## Overall Architecture

The overall architecture of TempoCL, with its core workflow as follows: (1) TASS first selects clean samples via a loss-based GMM, then labels noisy samples and separates closed-set from open-set noise by leveraging the absence of predefined categories for the latter. (2) MvTCL enforces same-scale consistency and different-scale complementarity to improve model robustness against label noise.

<p align="center">
<img src="./figures/RoRLNet.png" alt="RoRLNet" align=center />
</p>

## Datasets

### UEA 30 archive time series datasets

* [UEA 30 archive](http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip)

### Two individual large time series datasets

* [HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
* [ArabicDigits dataset](https://www.mustafabaydogan.com/research/time-series-data-mining/symbolic-representations-for-multivariate-time-series-classification-smts/)

## Usage
Install Pytorch and the necessary dependencies.

```
pip install -r requirements.txt
```

To train a RoRLNet model on a dataset, run

```bash
python main.py --archive UEA --dataset ArticularyWordRecognition --noise_type symmetric --label_noise_rate 0.2
```

## Citation

If you use this code for your research, please cite our paper:
```
@article{liu2026robust,
  title={Robust learning with time series noisy labels via self-supervised learning and soft labels refurbishment},
  author={Liu, Jiarong and Yang, Kaixiang and He, Jian and Yang, Chengrong and Yang, Shuang-hua and Zhou, Yujue},
  journal={Neurocomputing},
  pages={133059},
  year={2026},
  publisher={Elsevier}
}
```
