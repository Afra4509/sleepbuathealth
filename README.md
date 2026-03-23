<div align="center">

# 🌙 Sleep Quality Prediction AI

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A production-ready Machine Learning system that predicts sleep quality based on your lifestyle habits.*

[Overview](#-project-overview) • 
[Features](#-key-features) • 
[Dataset](#-dataset-insights) • 
[Visualizations](#-data-visualizations) • 
[Models](#-machine-learning-models) • 
[Installation](#-installation--usage)

</div>

---

## 📖 Project Overview

Welcome to the **Sleep Quality Prediction AI**! 

Sleep is the ultimate foundation of human health, directly influencing productivity, emotional resilience, and physical recovery. This repository provides a complete, end-to-end artificial intelligence framework that analyzes daily habits—such as *exercise frequency, daily steps, and stress levels*—to forecast an individual's **Quality of Sleep** (scored out of 10).

By recognizing these hidden health patterns, this tool helps users identify exactly which daytime routines they need to adjust to achieve perfect rest at night.

---

## ✨ Key Features

This project is built to true production-grade standards, bridging raw data into actionable insights:

- 🧠 **Advanced ML Pipeline:** Fully automated data extraction, imputation, scaling, and validation.
- 🧹 **Robust Preprocessing:** Handles missing values and dynamically encodes categorical lifestyle data natively.
- 🎯 **Automated Feature Selection:** Identifies and isolates the most impactful health factors utilizing regressor weights.
- 📈 **Hyperparameter Tuning:** Sweeps optimal model configurations using robust 5-Fold Cross Validation grids.
- 🖥️ **Interactive Desktop GUI:** A stunning, simple-to-use user application built with Tkinter.
- 📂 **CSV Batch Processing:** Evaluate hundreds of patients simultaneously via command-line integrations.

---

## 📊 Dataset Insights

This model is trained on a merged synergy of two extraordinarily comprehensive health datasets:
1. **`data111.csv`**: Broad lifestyle metrics including resting heart rates, step counts, and subjective stress levels.
2. **`Sleep_Efficiency.csv`**: Target-specific tracking including physical bedtimes, targeted sleep duration, and nightly awakenings.

### 🔍 Core Features Tracked
| Feature | Description |
| :--- | :--- |
| **Sleep Duration** | Total physical hours effectively slept. |
| **Stress Level** | Self-reported daily stress severity (Scale: 1-10). |
| **Heart Rate** | Resting heart beats per minute (BPM). |
| **Daily Steps** | Total active steps taken cumulatively per day. |
| **Awakenings** | Frequency of waking up mid-cycle during the night. |

<details>
<summary><b>Click to view a sample dataset matrix</b></summary>

| Sleep Duration | Stress Level | Heart Rate | Daily Steps | Sleep Efficiency | Awakenings | Quality of Sleep |
|----------------|--------------|-------------|-------------|------------------|------------|------------------|
| 6.5            | 7            | 75          | 5000        | 0.85             | 1.0        | 6.3              |
| 8.0            | 4            | 65          | 10000       | 0.95             | 0.0        | 9.4              |

</details>

---

## 📉 Data Visualizations

Understanding the model's mathematics is crucial! Below are the insights automatically extracted during our framework Exploratory Data Analysis (EDA).

### 1. Feature Importance
This chart illustrates which habits hold the heaviest mathematical weight in determining your rest. *(Notice how Stress Level and Sleep Duration completely dominate!)*  
<div align="center">
  <img src="https://raw.githubusercontent.com/Afra4509/sleepbuathealth/main/feature_importance.png" alt="Feature Importance" width="800"/>
</div>

### 2. Prediction vs Actual Validation
A tight clustering mapping directly along the center trendline proves our Gradient Boosting model's incredibly high accuracy verifying against unseen data.  
<div align="center">
  <img src="https://raw.githubusercontent.com/Afra4509/sleepbuathealth/main/prediction_vs_actual.png" alt="Prediction vs Actual" width="800"/>
</div>

### 3. Correlation Heatmap
A mathematical map displaying the direct linear relationships bounding every single combined lifestyle variable.  
<div align="center">
  <img src="https://raw.githubusercontent.com/Afra4509/sleepbuathealth/main/correlation_heatmap.png" alt="Correlation Heatmap" width="800"/>
</div>

### 4. Stress vs Quality
Highlights a highly explicit inverse relationship: as daily stress bounds increase, rest quality aggressively degrades.  
<div align="center">
  <img src="https://raw.githubusercontent.com/Afra4509/sleepbuathealth/main/stress_vs_quality.png" alt="Stress vs Quality" width="800"/>
</div>

### 5. Sleep Duration vs Quality
Shows a heavy positive scaling bound; securing sufficient hours in bed genuinely forms the absolute baseline for deep rest capabilities.  
<div align="center">
  <img src="https://raw.githubusercontent.com/Afra4509/sleepbuathealth/main/sleep_duration_vs_quality.png" alt="Sleep Duration vs Quality" width="800"/>
</div>

---

## 🤖 Machine Learning Models

Our master pipeline tracks and evaluates three unique algorithms simultaneously against the dataset variance:

1. **Linear Regression:** Standard testing baseline performance tracking.
2. **Random Forest Regressor:** A complex ensemble branching multiple internal decision tree outputs tracking subsets.
3. **Gradient Boosting Regressor 🏆:** Sequential optimization automatically minimizing absolute residual errors.

### 🏆 Final Output Benchmark (Gradient Boosting)
| Metric | Verification Score | Interpretation |
| :--- | :---: | :--- |
| **R² Score** | **0.9945** | The model accurately frames and accounts for **99.45%** of the dataset variance securely. |
| **MSE** | **0.0090** | Predictions deviate by less than a microscopic fractional bound from absolute reality! |

---

## 🏗️ Project Structure

```text
sleepbuathealth/
│
├── datasets/            # Raw combined training data & batch testing inputs
├── models/              # Verified serialized artifacts (best_model.pkl, scaler.pkl)
├── predict.py           # Core CLI Prediction utility (Single bindings & Batch scaling)
├── gui_predict.py       # Embedded Desktop Tkinter Graphical User Interface 
├── model_pipeline.py    # Master mathematical algorithm generation pipeline script 
├── requirements.txt     # Global repository Python environment dependency listings
└── README.md            # Project Operational Documentation Core
```

---

## 🚀 Installation & Usage

### 1. Installation
Deploy the AI securely into your local deployment machine:
```bash
git clone https://github.com/Afra4509/sleepbuathealth
cd sleepbuathealth
pip install -r requirements.txt
```

### 2. Using the Desktop Application (GUI)
Experience the model natively through our customized interface:
```bash
python gui_predict.py
```
> **How to use:** Enter your daily metrics into the respective prompt fields, click **Predict Sleep Quality**, and instantly receive your calculated score alongside generated personalized health recommendations!

### 3. Command Line Batch Predictions
Executing algorithmic bulk predictions scaling across hundreds of rows concurrently via CSVs:
```bash
python predict.py --csv datasets/input_data.csv
```
*Outputs are calculated and saved instantly to a new `batch_predictions.csv` trailing document.*

---

## 💡 Example CLI Prediction
**Programmatic Input Command:**
```bash
python predict.py --sleep_duration 6.5 --stress_level 7 --exercise_frequency 2 --heart_rate 75 --daily_steps 5000
```

**Evaluated Output Return:**
```text
Predicted Quality of Sleep: 6.3 / 10
```

---

## 🔮 Future Improvements
- ⌚ **Wearable API Integration:** Direct automated telemetry limits streaming from Oura Rings and Apple Watches.
- 📱 **Mobile Application Wrappers:** Porting the exact framework architectures explicitly into iOS Swift processing.
- 📡 **Live Dashboarding Servers:** Real-time WebSockets tracking integrations targeting ongoing telemetry.

---

## 📝 License
This exact project pipeline is completely open-source and universally licensed under the [**MIT License**](LICENSE). Feel free to pull, adapt, restructure, or evaluate the logic entirely without standard bounds.

---

<div align="center">
  <b>Developed with ❤️ by Afra</b><br>
  <i>Sleep Quality Prediction AI Platform System Framework</i>
</div>
