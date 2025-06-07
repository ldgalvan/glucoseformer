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

Key observation #1: The dataset contains continuous glucose monitoring (CGM) data collected every 5 minutes for 343 patients. A key observation was the wide variance in the number of glucose readings per patient—ranging from just 122 to over 30,000. This reflects differing durations of participation in the clinical trial.

<table>
  <tr>
    <td>

<strong>Key observation #1:</strong>  
The dataset contains continuous glucose monitoring (CGM) data collected every 5 minutes for 343 patients. A key observation was the wide variance in the number of glucose readings per patient—ranging from just 122 to over 30,000. This reflects differing durations of participation in the clinical trial.

<table>
  <tr><th>Patient ID</th><th>Number of Readings</th></tr>
  <tr><td>395</td><td>122</td></tr>
  <tr><td>259</td><td>140</td></tr>
  <tr><td>362</td><td>167</td></tr>
  <tr><td>146</td><td>184</td></tr>
  <tr><td>046</td><td>469</td></tr>
  <tr><td>348</td><td>626</td></tr>
  <tr><td>369</td><td>747</td></tr>
  <tr><td>076</td><td>1,042</td></tr>
  <tr><td>060</td><td>2,027</td></tr>
  <tr><td>...</td><td>...</td></tr>
  <tr><td>527</td><td>28,476</td></tr>
  <tr><td>062</td><td>28,457</td></tr>
  <tr><td>030</td><td>28,450</td></tr>
  <tr><td>511</td><td>28,378</td></tr>
</table>

To avoid potential overfitting to patients with more abundant data, we chose a non-overlapping scheme in processing our sequences.

    </td>
    <td>
      <img src="patient_counts_distribution.png" width="400">
    </td>
  </tr>
</table>

To avoid potential overfitting to the patient's with more abundant data, we chose a non-overlapping scheme in processing our sequences.

Key observation #2: Both Basal and Bolus insulin dose amounts and frequency were reported in the dataset as well. To visualize this, we can plot vertical lines over every bolus/basal event

![CGM data](cgm_patient_81_day_plot.png)

