import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os

def load_artifacts():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        selector = joblib.load('selector.pkl')
        return model, scaler, imputer, selector
    except FileNotFoundError as e:
        print(f"Error: {e}.\nMake sure you are in the directory containing the .pkl files.")
        sys.exit(1)

def run_prediction(data_dict, model, scaler, imputer, selector):
    original_features = selector['features']
    feature_mask = selector['mask']

    # Initialize dataframe with all NaNs for exactly matching the original pipeline shapes
    input_df = pd.DataFrame(columns=original_features)
    input_df.loc[0] = [np.nan] * len(original_features)

    # Safely assign provided values into the correctly mapped columns
    for key, value in data_dict.items():
        if key in original_features and pd.notna(value) and value is not None:
            input_df[key] = float(value)

    # 1. Imputation
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=original_features)
    
    # 2. Scaling
    input_scaled = scaler.transform(input_imputed)
    
    # 3. Feature Selection Masking
    input_selected = input_scaled[:, feature_mask]

    # Predict
    prediction = model.predict(input_selected)[0]
    return prediction

def main():
    parser = argparse.ArgumentParser(description="Production prediction interface for Quality of Sleep.")
    
    # Batch Predict Option
    parser.add_argument('--csv', type=str, help="Path to input CSV for batch prediction processing.")
    
    # Individual Predict Options
    parser.add_argument('--sleep_duration', type=float, help="Sleep Duration (hours, e.g., 6.5)")
    parser.add_argument('--stress_level', type=float, help="Stress Level (scale, e.g., 7)")
    parser.add_argument('--exercise_frequency', type=float, help="Exercise frequency (days/week, e.g., 2)")
    parser.add_argument('--heart_rate', type=float, help="Heart Rate (bpm, e.g., 75)")
    parser.add_argument('--daily_steps', type=float, help="Daily Steps (count, e.g., 5000)")
    parser.add_argument('--sleep_efficiency', type=float, help="Sleep efficiency (ratio, e.g., 0.8)")
    parser.add_argument('--awakenings', type=float, help="Number of awakenings (count, e.g., 1)")

    args = parser.parse_args()

    # Bootstrap Artifacts
    model, scaler, imputer, selector = load_artifacts()

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"File {args.csv} does not exist.")
            sys.exit(1)
            
        print(f"Processing batch predictions utilizing {args.csv}...")
        try:
            df = pd.read_csv(args.csv)
            predictions = []
            for _, row in df.iterrows():
                # Map available columns strictly mapping identically
                pred = run_prediction(row.to_dict(), model, scaler, imputer, selector)
                predictions.append(pred)
                
            df['Predicted Quality of Sleep'] = predictions
            out_file = 'batch_predictions.csv'
            df.to_csv(out_file, index=False)
            print(f"Successfully processed {len(df)} predictions. Saved to '{out_file}'.")
            
        except Exception as e:
            print(f"Failed to process CSV file due to: {e}")
            sys.exit(1)
            
    else:
        # Evaluate Single Input Case
        inputs = {
            'Sleep Duration': args.sleep_duration,
            'Stress Level': args.stress_level,
            'Exercise frequency': args.exercise_frequency,
            'Heart Rate': args.heart_rate,
            'Daily Steps': args.daily_steps,
            'Sleep efficiency': args.sleep_efficiency,
            'Awakenings': args.awakenings
        }
        
        if all(v is None for v in inputs.values()):
            parser.print_help()
            print("\nError: You must provide at least one parameter natively or target a file using --csv.")
            sys.exit(1)

        pred = run_prediction(inputs, model, scaler, imputer, selector)
        print(f"Predicted Quality of Sleep: {pred:.1f} / 10")

if __name__ == "__main__":
    main()
