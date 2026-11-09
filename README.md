# Glucoformer: Multimodal Cross-subject Glucose Forecasting with Long-term Prediction Capability

<img src="figures/Fig. 1 Overview of Glucoformer.png" alt="Overview of Glucoformer" />

> Paper: 
>
> Authors: Ruixi Chen, Jia Long, Yuxuan Liao, Jiyun Zheng, Gaoxiang Li, Ao Gao, Hongmei Lu, Zhimin Zhang

---

## Table of Contents:
- [1. Project Contributions](#1-Project-Contributions)
- [2. Project Folder Structure](#2-Project-Folder-Structure)
- [3. Environment Setup](#3-Environment-Setup)
- [4. Dataset Preparation](#4-Dataset-Preparation)
- [5. Contact](#5-Contact)
---

## 1. Project Contributions
**Glucoformer** is a novel model for long-term blood glucose (BG) forecasting, achieving state-of-the-art (SOTA) predictive accuracy and computational efficiency. Built upon an enhanced Crossformer architecture, Glucoformer introduces several key innovations:

- **Statistical-Aware Segment Embedding (SSE):**  
  Glucoformer extends conventional segment embedding by incorporating segment-level statistical information, including mean and standard deviation. Combined with a lightweight gating mechanism and positional encoding, this module enhances local distribution awareness while preserving temporal and cross-dimensional dependencies.

- **Enhanced Two-Stage Attention (ETSA):**  
  The proposed ETSA improves feature representation across both temporal and cross-dimensional spaces. In the temporal stage, Patch Linear Attention (PLA) with Depthwise Separable Convolution (DWConv) efficiently captures long-range dependencies. In the cross-dimension stage, a context-aware adaptive routing strategy dynamically refines router tokens to better model inter-variable relationships.

- **Lightweight Decoder Design:**  
  The original Two-Stage Attention (TSA) layer in the decoder is removed to reduce redundant computation. This simplified decoder improves computational efficiency while further enhancing predictive performance.

- **Comprehensive Evaluation and Analysis:**  
  Glucoformer is extensively evaluated on three datasets: T1DMS, OhioT1DM, and DiaTrend. Experiments include direct comparisons with established and SOTA BG forecasting models (GRU, LSTM, Transformer, Informer, DLinear, TimeXer, PatchTST, and Crossformer), along with long-horizon forecasting analysis, clinical accuracy and safety evaluation, ablation studies, interpretability analysis, and computational complexity profiling. Additionally, zero-shot transfer evaluation on the external DiaTrend dataset further validates its cross-domain generalization capability.


## 2. Project Folder Structure

<pre>
Glucoformer
├── data_provider/
│   ├── DiaTrend_Dataset.py
│   ├── OhioT1DM_Dataset.py
│   ├── T1DMS_Dataset.py
│   └── timefeatures.py
├── exp/
│   ├── loss.py
│   └── train_eval.py
├── models/
│   ├── Crossformer/
│   ├── DLinear/
│   ├── Glucoformer/
│   │   ├── attn.py
│   │   ├── cross_decoder.py
│   │   ├── cross_embed.py
│   │   ├── cross_encoder.py
│   │   ├── Glucoformer_parameters.py
│   │   └── Glucoformer.py
│   ├── GRU/
│   ├── Informer/
│   ├── LSTM/
│   ├── PatchTST/
│   ├── TimeXer/
│   └── Transformer/
├── utils/
│   ├── global_options.py
│   └── tools.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
</pre>


## 3. Environment Setup

To set up the environment for this project, please follow these steps:

### (1)Clone the repository

```bash
git clone https://github.com/CSU-RuixiChen/Glucoformer.git
cd Glucoformer
```

### (2)Create and activate a Python virtual environment

```bash
# Using venv
python3 -m venv ENVIRONMENT_NAME
source ENVIRONMENT_NAME/bin/activate

# Or using conda
conda create -n ENVIRONMENT_NAME python=3.13.5
conda activate ENVIRONMENT_NAME
```

### (3)Install required packages

```bash
pip install -r requirements.txt
```

After these steps, your environment will be ready to run the code and experiments in this repository.


## 4. Dataset Preparation

Three datasets were utilized in this project: the **T1DMS virtual dataset**, the **OhioT1DM real-world dataset**, and the **DiaTrend real-world dataset**.

The Glucoformer model is first pre-trained on the T1DMS dataset to learn general glucose regulation patterns from large-scale virtual subjects. It is then fine-tuned on the more heterogeneous OhioT1DM dataset to improve adaptation to real-world glucose dynamics. The held-out OhioT1DM test set is used to evaluate predictive performance on unseen subjects. In addition, the DiaTrend dataset is employed for zero-shot inference to further assess the cross-subject generalization capability of Glucoformer on an entirely independent external dataset.

Due to file size and data sharing restrictions, the original datasets are not included in this repository. However, the complete datasets can be obtained from the following official sources:

- **T1DMS Virtual Dataset**: The T1DMS dataset was generated using the UVa/Padova simulator (available at https://tegvirginia.com/t1dms/), and we provide scenario files for scenarios 1–60 to facilitate replication.
- **OhioT1DM Real Dataset**: https://webpages.charlotte.edu/rbunescu/ohiot1dm.html
- **DiaTrend Real Dataset**: The DiaTrend dataset is accessible via www.synapse.org after account registration, certification, submission of a data use statement, and agreement to the dataset usage terms.

After obtaining the datasets, please place them in the corresponding subfolders following the directory structures below.

- **T1DMS Raw Dataset:**

  <pre>
  Glucose_Data/T1DMS_Dataset/
  ├── sim_data_train_dataset.mat
  ├── sim_data_validate_dataset.mat
  ├── sim_results_train_dataset.mat
  └── sim_results_validate_dataset.mat
  </pre>

- **OhioT1DM Raw Dataset:**

  <pre>
  Glucose_Data/OhioT1DM_Dataset/OhioT1DM_raw_dataset
  ├── OhioT1DM-2018-training/
  │   ├── 559-ws-training.xml
  │   ├── 563-ws-training.xml
  │   ├── 570-ws-training.xml
  │   ├── 575-ws-training.xml
  │   ├── 588-ws-training.xml
  │   └── 591-ws-training.xml
  ├── OhioT1DM-2018-testing/
  │   ├── 559-ws-testing.xml
  │   ├── 563-ws-testing.xml
  │   ├── 570-ws-testing.xml
  │   ├── 575-ws-testing.xml
  │   ├── 588-ws-testing.xml
  │   └── 591-ws-testing.xml
  ├── OhioT1DM-2020-training/
  │   ├── 540-ws-training.xml
  │   ├── 544-ws-training.xml
  │   ├── 552-ws-training.xml
  │   ├── 567-ws-training.xml
  │   ├── 584-ws-training.xml
  │   └── 596-ws-training.xml
  └── OhioT1DM-2020-testing/
      ├── 540-ws-testing.xml
      ├── 544-ws-testing.xml
      ├── 552-ws-testing.xml
      ├── 567-ws-testing.xml
      ├── 584-ws-testing.xml
      └── 596-ws-testing.xml
  </pre>

- **DiaTrend Raw Dataset:**

  <pre>
  Glucose_Data/DiaTrend_Dataset/
  └── Raw/
      ├── Subject29.xlsx
      ├── Subject30.xlsx
      ├── Subject31.xlsx
      ├── Subject36.xlsx
      ├── Subject37.xlsx
      ├── Subject38.xlsx
      ├── Subject39.xlsx
      ├── Subject42.xlsx
      ├── Subject45.xlsx
      ├── Subject46.xlsx
      ├── Subject47.xlsx
      ├── Subject49.xlsx
      ├── Subject50.xlsx
      ├── Subject51.xlsx
      ├── Subject52.xlsx
      ├── Subject53.xlsx
      └── Subject54.xlsx
  </pre>


## 4. Contact
- **Zhimin Zhang** at [zmzhang@csu.edu.cn](zmzhang@csu.edu.cn)
- **Ruixi Chen** at [242311059@csu.edu.cn](242311059@csu.edu.cn)

