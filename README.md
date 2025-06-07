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

The dataset includes continuous glucose monitoring (CGM) data recorded every 5 minutes for a cohort of 343 patients. One of the most notable characteristics is the wide variability in the number of CGM readings per patient, ranging from just 122 to over 28,000 entries. This variance reflects the differing durations of patient participation in the clinical trial.

To better understand this distribution, we visualize the number of CGM measurements per patient below:
![CGM data](patient_counts_distribution.png)

This distribution reveals a diverse set of data points across patients. However, to mitigate the risk of overfitting to patients with disproportionately large data volumes, we adopted a non-overlapping window strategy when constructing model input sequences.

In addition to glucose measurements, the dataset also includes detailed records of Basal and Bolus insulin deliveries—both in terms of dosage and frequency. To visualize these clinical interventions, we plot vertical markers over time, corresponding to each insulin delivery event in a sample patient’s timeline.

![CGM data](cgm_patient_81_day_plot.png)

## Feature Engineering

With frequent bolus injections happening, we introduce a new variable which incorporated the decay rate of Insulin on Board (IOB). This helps our time-series transformer learn adapt and account for this variable, as it is something which directly impacts future CGM measurments.
