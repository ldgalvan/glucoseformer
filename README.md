# GlucoseFormer 📈


![Model Forecast Comparison](sample_27_forecast_dark.png)

## Overview

GlucoseFormer is a time-series Transformer-based model designed to forecast continuous glucose monitoring (CGM) levels. To enhance prediction accuracy and interpretability, the model incorporates:

- **[RoPE (Rotary Positional Encoding)](https://arxiv.org/abs/2104.09864)**  
  Improves the model’s ability to understand time-based relationships by encoding relative position information—ideal for long context windows.

- **IOB (Insulin on Board)**  
  Simulates the decaying effect of bolus insulin over time, helping the model factor in insulin that is still active in the body.

- **PCA (Principal Component Analysis)**  
  Reduces the dimensionality of input features while preserving critical information, which improves generalization and training speed.

## Dataset

This project uses the publicly available dataset from:

**[The Insulin-Only Bionic Pancreas Pivotal Trial: Testing the iLet in Adults and Children with Type 1 Diabetes](https://public.jaeb.org/datasets/diabetes)**  
Hosted by the JAEB Center for Health Research.

---

## Data analysis

The dataset contains continuous glucose monitoring (CGM) data collected every 5 minutes for 343 patients. A key observation was the wide variance in the number of glucose readings per patient—ranging from just 122 to over 30,000. This reflects differing durations of participation in the clinical trial.

| Patient ID | Number of Readings |
| ---------- | ------------------ |
| 395        | 122                |
| 259        | 140                |
| 362        | 167                |
| 146        | 184                |
| 046        | 469                |
| 348        | 626                |
| 369        | 747                |
| 076        | 1,042              |
| 060        | 2,027              |
| 507        | 4,025              |
| 531        | 4,287              |
| 357        | 4,317              |
| 553        | 4,335              |
| 317        | 5,033              |
| 020        | 5,439              |
| 411        | 5,891              |
| 406        | 7,717              |
| 580        | 8,881              |
| 444        | 10,342             |
| 314        | 10,695             |


| Patient ID | Number of Readings |
| ---------- | ------------------ |
| 081        | 30,289             |
| 464        | 29,669             |
| 009        | 29,662             |
| 584        | 29,066             |
| 273        | 29,042             |
| 535        | 28,923             |
| 492        | 28,874             |
| 429        | 28,833             |
| 309        | 28,791             |
| 588        | 28,767             |
| 050        | 28,529             |
| 069        | 28,529             |
| 383        | 28,508             |
| 359        | 28,508             |
| 529        | 28,484             |
| 139        | 28,482             |
| 527        | 28,476             |
| 062        | 28,457             |
| 030        | 28,450             |
| 511        | 28,378             |

