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

The dataset contains cgm monitoring every 5 minutes for 343 patients' glucose levels. The first thing of importance was the amount of measurements per patient. these ranged from 122 to 28378, indicating some patients stayed in the group for varying amounts of time. 


+----+-----+
|PtID|count|
+----+-----+
| 395|  122|
| 259|  140|
| 362|  167|
| 146|  184|
|  46|  469|
| 348|  626|
| 369|  747|
|  76| 1042|
|  60| 2027|
| 507| 4025|
| 531| 4287|
| 357| 4317|
| 553| 4335|
| 317| 5033|
|  20| 5439|
| 411| 5891|
| 406| 7717|
| 580| 8881|
| 444|10342|
| 314|10695|
+----+-----+

+----+-----+
|PtID|count|
+----+-----+
|  81|30289|
| 464|29669|
|   9|29662|
| 584|29066|
| 273|29042|
| 535|28923|
| 492|28874|
| 429|28833|
| 309|28791|
| 588|28767|
|  50|28529|
|  69|28529|
| 383|28508|
| 359|28508|
| 529|28484|
| 139|28482|
| 527|28476|
|  62|28457|
|  30|28450|
| 511|28378|
+----+-----+
