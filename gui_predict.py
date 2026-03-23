import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import joblib
import sys
import os
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class SleepQualityPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sleep Quality Predictor")
        self.root.geometry("450x850")
        self.root.configure(padx=25, pady=25)
        
        # Load artifacts immediately
        try:
            self.model = joblib.load('best_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.imputer = joblib.load('imputer.pkl')
            self.selector = joblib.load('selector.pkl')
            
            self.original_features = self.selector['features']
            self.feature_mask = self.selector['mask']
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Model files could not be loaded.\nEnsure the .pkl files are available locally.\n\nError: {e}")
            self.root.destroy()
            return
            
        self.create_widgets()
        
    def create_widgets(self):
        # Main Header Title
        title_label = tk.Label(self.root, text="Sleep Quality Predictor", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=(0, 15))
        
        self.entries = {}
        
        # Define fields mapped to underlying feature processing keys
        fields = [
            ("Sleep Duration (hours)", "Sleep Duration"),
            ("Stress Level (1-10)", "Stress Level"),
            ("Exercise Frequency (per week)", "Exercise frequency"),
            ("Heart Rate (bpm)", "Heart Rate"),
            ("Daily Steps", "Daily Steps"),
            ("Sleep Efficiency (0-1)", "Sleep efficiency"),
            ("Awakenings (number)", "Awakenings")
        ]
        
        for label_text, feature_name in fields:
            frame = tk.Frame(self.root)
            frame.pack(fill="x", pady=5)
            
            lbl = tk.Label(frame, text=label_text, width=28, anchor="w", font=("Helvetica", 10))
            lbl.pack(side="left")
            
            ent = tk.Entry(frame, width=15, font=("Helvetica", 11))
            ent.pack(side="right")
            
            self.entries[feature_name] = ent
            
        # Target Predict Button
        self.predict_btn = tk.Button(self.root, text="Predict Sleep Quality", command=self.predict_quality, 
                                     bg="#007BFF", fg="white", font=("Helvetica", 12, "bold"), cursor="hand2")
        self.predict_btn.pack(pady=15, fill="x", ipady=5)
        
        # Output Resolution Text
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 15, "bold"))
        self.result_label.pack()
        
        self.confidence_label = tk.Label(self.root, text="", font=("Helvetica", 12, "italic"))
        self.confidence_label.pack(pady=5)
        
        self.recommendation_label = tk.Label(self.root, text="", font=("Helvetica", 11), justify="center", wraplength=400)
        self.recommendation_label.pack(pady=5)
        
        # Matplotlib Chart Frame
        self.chart_frame = tk.Frame(self.root)
        self.chart_frame.pack(pady=5, fill="both", expand=True)
        
        self.fig = Figure(figsize=(4, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def predict_quality(self):
        # 1. Validation and Collection
        data_dict = {}
        empty_count = 0
        
        for feature_name, ent in self.entries.items():
            val = ent.get().strip()
            if val == "":
                data_dict[feature_name] = np.nan
                empty_count += 1
            else:
                try:
                    data_dict[feature_name] = float(val)
                except ValueError:
                    messagebox.showwarning("Invalid Input", f"Please enter a valid numerical value for '{feature_name}'.")
                    return
        
        if empty_count == len(self.entries):
            messagebox.showwarning("Incomplete Data", "Please generate at least one input feature before querying prediction.")
            return

        # 2. Re-Shape Processing
        input_data = pd.DataFrame(columns=self.original_features)
        input_data.loc[0] = [np.nan] * len(self.original_features)
        
        # Emplace target values cleanly
        for k, v in data_dict.items():
            if k in self.original_features:
                input_data[k] = v
                
        try:
            # 3. Model Flow Execution
            input_imputed = pd.DataFrame(self.imputer.transform(input_data), columns=self.original_features)
            input_scaled = self.scaler.transform(input_imputed)
            
            input_selected = input_scaled[:, self.feature_mask]
            
            prediction = self.model.predict(input_selected)[0]
            
            # Constrain UI bounds intelligently 0 -> 10 mapping format requested
            prediction_clamp = max(0.0, min(10.0, prediction))
            
            # Formulate Output
            self.result_label.config(text=f"Predicted Quality of Sleep: {prediction_clamp:.1f} / 10")
            
            if prediction_clamp < 5.0:
                msg = "Your predicted sleep quality is POOR ⚠️"
                color = "#DC3545" # Red
                rec_text = "Recommendations:\n- Increase sleep duration\n- Reduce stress level\n- Improve sleep efficiency\n"
            elif prediction_clamp <= 7.0:
                msg = "Your predicted sleep quality is FAIR 😐"
                color = "#FD7E14" # Orange
                rec_text = "Recommendations:\n- Consider moderate adjustments to bedtime routine\n- Monitor daily activity to ease evening relaxation\n"
            else:
                msg = "Your predicted sleep quality is EXCELLENT 🎉"
                color = "#28A745" # Green
                rec_text = "Recommendations:\n- Great job! Maintain your current habits\n- Continue your positive lifestyle routines\n"
                
            self.confidence_label.config(text=msg, fg=color)
            self.recommendation_label.config(text=rec_text)

            # 4. Update Matplotlib Chart
            self.ax.clear()
            bars = self.ax.bar(['Sleep Quality'], [prediction_clamp], color=color)
            self.ax.set_ylim(0, 10)
            self.ax.set_ylabel('Score (0-10)')
            self.ax.set_title('Predicted Output Scale')
            
            # Add text on top of bar
            for bar in bars:
                yval = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{prediction_clamp:.1f}', ha='center', va='bottom', fontweight='bold')
                
            self.canvas.draw()

            # 5. Save to History CSV
            history_file = 'prediction_history.csv'
            history_record = data_dict.copy()
            history_record['Predicted Quality of Sleep'] = prediction_clamp
            history_record['Timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Standardize columns ordering logically across missing arrays
            df_hist = pd.DataFrame([history_record])
            if not os.path.isfile(history_file):
                df_hist.to_csv(history_file, index=False)
            else:
                df_hist.to_csv(history_file, mode='a', header=False, index=False)
            
        except Exception as e:
            messagebox.showerror("Execution Error", f"The model encountered an error calculating prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SleepQualityPredictorApp(root)
    root.mainloop()
