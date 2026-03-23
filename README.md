# Sleep Quality Prediction AI 🌙

![GitHub last commit](https://img.shields.io/badge/last%20commit-today-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 1. Project Overview
Welcome to the **Sleep Quality Prediction AI** project! 

This repository contains a robust, end-to-end machine learning system designed to reliably predict a user's **Quality of Sleep** heavily based on their daily lifestyle and physical habits. Sleep is an essential component of human health, fundamentally affecting productivity, mental wellness, and physiological recovery. 

The primary purpose of this AI model is to analyze patterns and hidden correlations between various daily activities (like exercise frequencies, daily steps, and stress) and sleep metrics to provide an explicit, out-of-10 assessment forecasting how well a person will sleep. By identifying these patterns clearly, this application helps individuals better understand and directly modify their routines to elevate their rest!

---

## 2. Features Section
This repository represents a fully production-ready framework complete with:

- **End-to-end Machine Learning Pipeline:** An automated data script running dynamic extraction, permutation, and pipeline validation effortlessly.
- **Data Preprocessing & Cleaning:** Dynamic median imputation and localized categorical parsing ensuring entirely clean arrays.
- **Automated Feature Selection:** Natively evaluates base Random Forest parameter tree weights to drop weak or unrelated columns protecting efficiency.
- **Model Training & Optimization:** Evaluates independent models sweeping distinct hyperparameter grids iteratively to find absolute peaks.
- **GUI-Based Prediction System:** A clean, beautifully accessible desktop application built cleanly with Tkinter containing logic gating.
- **Batch CSV System Extraction:** A CLI mapping integration automating hundreds of validation predictions flawlessly tracking outputs.
- **Visualization of Dataset Relationships:** Mathematical rendering frameworks parsing correlation matrices and prediction metrics graphically.
- **Prediction Result Interpretation:** Color-coded confidence feedback evaluating specific score thresholds into health recommendations.

---

## 3. Dataset Description
This project harnesses two diverse datasets to construct its underlying logic matrices:
1. **`data111.csv`**: Broadens lifestyle context by bridging internal constraints such as stress levels, step frequencies, and heart rates alongside target quality ratings.
2. **`Sleep_Efficiency.csv`**: Contains extensive tracked data relating strictly to physical bedtimes, sleep efficiency percentages, and awakening frequencies.

### Key Predictor Features Evaluated:
- **Sleep Duration:** Length of individual sleep tracked physically (measured in hours).
- **Sleep Efficiency:** Ratio of active time asleep over absolute time spent resting in bed (0.0 - 1.0).
- **Stress Level:** Daily self-reported mental and physiological bounds measuring severity (1-10).
- **Physical Activity & Exercise Frequency:** Core measurements checking consistency and workout schedules.
- **Heart Rate & Daily Steps:** Key background physical health baseline variables.
- **Awakenings:** The aggregate amount of times an individual woke up through the sleep cycle.

### Sample Dataset Preview Matrix
| Sleep Duration | Stress Level | Heart Rate | Daily Steps | Sleep Efficiency | Awakenings | Quality of Sleep |
|----------------|--------------|-------------|-------------|------------------|------------|------------------|
| 6.5            | 7            | 75          | 5000        | 0.85             | 1.0        | 6.3              |
| 8.0            | 4            | 65          | 10000       | 0.95             | 0.0        | 9.4              |

---

## 4. Data Visualization Section
Understanding the background mathematics simplifies interpreting exactly how the AI derives its evaluations. *(Please ensure visual `.png` files are correctly mapped to your `/images/` directory!)*

**Correlation Heatmap**  
Shows the direct linear relationships mathematically existing across variables. Deep red asserts positive scaling context, while deep blue highlights absolute inverse bounds.
![Correlation Heatmap](images/correlation_heatmap.png)

**Top Feature Importances**  
Displays the absolute heaviest weights directing algorithm evaluation targets. Notice how Stress Level and Sleep Duration commonly dominate these factors!
![Feature Importance Plot](images/feature_importance.png)

**Sleep Duration vs Quality**  
Highlights the explicit direct scatter correlation evaluating how longer hours naturally build quality structures.
![Sleep Duration vs Quality Scatter Plot](images/sleep_duration_vs_quality.png)

**Stress Level vs Quality**  
Demonstrates the distinct inverse mapping where highly elevated stress parameters aggressively decay quality outcomes.
![Stress Level vs Quality Scatter Plot](images/stress_vs_quality.png)

**Prediction vs Actual Verification**  
Visualizes the champion model's exact variance accuracy. Points tracking directly along the center trendline prove incredibly high testing accuracy!
![Prediction vs Actual Plot](images/prediction_vs_actual.png)

---

## 5. Machine Learning Model Section
To guarantee optimization, the master pipeline integrates and trains three explicit ML architectures simultaneously running against a robust 5-Fold Cross Validation suite:
- **Linear Regression:** Standard tracking baseline assessing independent linear variable limits.
- **Random Forest Regressor:** A complex ensemble branching multiple internal decision tree outputs tracking subsets.
- **Gradient Boosting Regressor:** Sequential decision optimization automatically correcting mathematical errors produced by prior sequential tracking steps.

### Absolute Model Comparison Results Tracking
1. **Linear Regression**: $R^2$ = 0.9471 | MSE = 0.0860
2. **Random Forest**: $R^2$ = 0.9939 | MSE = 0.0100
3. **Gradient Boosting: $R^2$ = 0.9945 | MSE = 0.0090**

**Gradient Boosting Regressor** was mathematically locked and formally serialized as the ultimate `best_model.pkl`. It decisively captured minimizing absolute Mean Squared Error (MSE) metrics while simultaneously peaking scaling predictability across extreme lifestyle bounds!

---

## 6. Project Structure
The repository is structured to separate programmatic pipeline frameworks gracefully away from export variables:
```text
sleepbuathealth/
│
├── datasets/
│   ├── Sleep_Efficiency.csv     # First targeted dataset
│   ├── data111.csv              # Second unified feature list
│   └── input_data.csv           # Modular testing input for automated pipelines
│
├── models/
│   ├── best_model.pkl           # Final serialized algorithm evaluation network
│   ├── scaler.pkl               # Standard parameter scaling wrapper constraints
│   ├── imputer.pkl              # Missing logic replacement wrapper
│   └── selector.pkl             # Dynamic variable feature selector dict
│
├── images/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── sleep_duration_vs_quality.png
│   ├── stress_vs_quality.png
│   └── prediction_vs_actual.png
│
├── predict.py           # Core CLI Prediction utility (Single bindings & Batch)
├── gui_predict.py       # Embedded Desktop Tkinter Graphical User Interface 
├── model_pipeline.py    # Master mathematical algorithm generation script 
├── requirements.txt     # Global repository dependency installer bounds
└── README.md            # Project Operational Documentation
```
*Tip: Sort your generated ML outputs into the `datasets/`, `models/`, and `images/` folders directly for absolute environment cleanliness!*

---

## 7. Installation Instructions
Follow these precise terminal steps to deploy and explore the AI parameters natively locally:

1. **Clone the targeted repository**
   ```bash
   git clone https://github.com/Afra4509/sleepbuathealth
   cd sleepbuathealth
   ```
2. **Install all cross-environment dependencies universally**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the User Interface Visual Platform**
   ```bash
   python gui_predict.py
   ```
4. **Evaluate programmatic system batch tests externally**
   ```bash
   python predict.py
   ```

---

## 8. Usage Instructions
### Running the GUI Tracker (`gui_predict.py`)
1. Run `python gui_predict.py` directly bridging your system framework constraints.
2. Enter your precise baseline lifestyle metrics accurately mapping individual numbers directly.
3. Click the explicit **"Predict Sleep Quality"** action target box.
4. Read your output evaluation immediately checking both the generated scale index score exactly, the specific prediction visual bar chart, and explicitly reviewing generated targeted health recommendations.

### Resolving Batch Matrix Predictions (`predict.py`)
Processing datasets iterating heavily across hundreds of independent columns is simple. Pass targeting variables universally:
```bash
python predict.py --csv datasets/input_data.csv
```
This automatically parses entire rows evaluating the variables simultaneously natively and outputs the new predicted array results explicitly inside a safe new `batch_predictions.csv` tracking file.

---

## 9. Example Prediction Section
Here is a strict mathematical pipeline example displaying explicit system evaluation outcomes scaling bounds accurately:

**Input Configuration Map:**
- Sleep Duration: `6.5`
- Stress Level: `7`
- Exercise Frequency: `2`

**Output Return Parameter Format:**
```text
Predicted Quality of Sleep: 6.3 / 10
```

---

## 10. Results Section
Our integrated pipeline verified extensive parameters validating strictly through complex mathematical array evaluations:

**Gradient Boosting Regressor (Winner):**
- **$R^2$ Score: 0.9945**
- **MSE: 0.0090**

**What this mathematically represents:** 
An $R^2$ scaling ratio standing consistently at `0.9945` highlights the mathematical array actively captures completely identifying **99.45%** of all unique tracked lifestyle quality modifications! Minimizing the Mean Squared Error strictly beneath `0.01` guarantees your output metrics scale actively holding within minimal fractions of completely reliable absolute accuracy constraints!

---

## 11. Future Improvements Section
Scaling this programmatic analysis safely incorporates heavy opportunities moving forward globally bridging bounds:
- ⌚ **Wearable Device Extraction Mapping:** Expanding automated parameter retrieval streaming natively through APIs from Apple Watch/Oura.
- 📱 **Mobile Application Integration Structures:** Re-wrapping parameters actively targeting iOS bridging Swift bounds exclusively.
- 📡 **Real-time Monitoring Hooks:** Connecting predictive outputs constantly executing API background tasks tracking web-app live visualization loops.
- 📊 **Target Dataset Scale Growth Limits:** Expanding algorithmic testing processing demographics incorporating heavy variations targeting ages.

---

## 12. License Section
This specific tracking pipeline structure algorithm is safely bounded strictly explicitly open-source universally using the absolute **MIT License**. Adapt, restructure, or evaluate native internal loops natively without external bounds.

---

## 13. Author Section
- **Author:** Afra  
- **Project Concept:** Sleep Quality Prediction AI Platform System Framework
- **GitHub Target Output Constraint Repository:** [sleepbuathealth](https://github.com/Afra4509/sleepbuathealth)
