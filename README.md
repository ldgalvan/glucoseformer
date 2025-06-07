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

Key observation #1: The dataset contains continuous glucose monitoring (CGM) data collected every 5 minutes for 343 patients. A key observation was the wide variance in the number of glucose readings per patient—ranging from just 122 to over 28,378. This reflects differing durations of participation in the clinical trial. To furthe diagnose this, we plot patients vs total cgm measurements/patient

![CGM data](patient_counts_distribution.png)

From this chart, we can see there is a fair amount of diverse data. Still, to avoid potential overfitting to patients with more abundant data, we chose a non-overlapping scheme in processing our sequences.

Key observation #2: Both Basal and Bolus insulin dose amounts and frequency were reported in the dataset as well. To visualize this, we can plot vertical lines over every bolus/basal event

![CGM data](cgm_patient_81_day_plot.png)

