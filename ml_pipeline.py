import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Load datasets
    df1 = pd.read_csv('Sleep_Efficiency.csv')
    df2 = pd.read_csv('data111.csv')

    # Rename to match
    if 'Sleep duration' in df1.columns:
        df1 = df1.rename(columns={'Sleep duration': 'Sleep Duration'})

    # Pre-calculate approximate Quality of Sleep for df1 using Sleep efficiency to preserve data
    if 'Sleep efficiency' in df1.columns:
        df1['Quality of Sleep'] = df1['Sleep efficiency'] * 10
    
    # Concatenate datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # 2. Clean the data
    # Convert categorical columns
    categorical_cols = ['Gender', 'Occupation', 'BMI Category']
    for col in categorical_cols:
        if col in df.columns:
            # Map strings to int codes, replacing missing (-1) with NaN
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes.replace(-1, np.nan)

    # 4. Feature engineering explicitly requested
    features = ['Sleep Duration', 'Sleep efficiency', 'Stress Level', 
                'Physical Activity Level', 'Heart Rate', 'Daily Steps', 
                'Awakenings', 'Exercise frequency', 'Gender', 'Occupation', 'BMI Category']
    target = 'Quality of Sleep'

    # Filter columns to only what is needed
    all_cols = features + [target]
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_subset = df[all_cols]

    # Handle missing values via simple imputation
    imputer = SimpleImputer(strategy='median')
    df_imputed = df_subset.copy()
    df_imputed[features] = imputer.fit_transform(df_subset[features])
    df_imputed[target] = df_imputed[target].fillna(df_imputed[target].median())

    # 3. Perform EDA
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Sleep Duration vs Quality of Sleep plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Sleep Duration', y='Quality of Sleep', data=df_imputed, color='blue', alpha=0.6)
    plt.title('Sleep Duration vs Quality of Sleep')
    plt.savefig('sleep_duration_vs_quality.png')
    plt.close()

    # Stress Level vs Quality of Sleep plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Stress Level', y='Quality of Sleep', data=df_imputed, color='red', alpha=0.6)
    plt.title('Stress Level vs Quality of Sleep')
    plt.savefig('stress_vs_quality.png')
    plt.close()

    # 5. Normalize numerical data
    X = df_imputed[features]
    y = df_imputed[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 6. Train two models
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # 7. Evaluate models
    print("Linear Regression Metrics:")
    print(f"MSE: {mean_squared_error(y_test, lr_preds):.4f}")
    print(f"R2 Score: {r2_score(y_test, lr_preds):.4f}\n")

    print("Random Forest Regressor Metrics:")
    print(f"MSE: {mean_squared_error(y_test, rf_preds):.4f}")
    print(f"R2 Score: {r2_score(y_test, rf_preds):.4f}\n")

    # 8. Save models
    joblib.dump(lr_model, 'linear_regression_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')

    # Write evaluation metrics to a text file
    with open('metrics_output.txt', 'w') as f:
        f.write("--- Linear Regression ---\n")
        f.write(f"MSE: {mean_squared_error(y_test, lr_preds):.4f}\n")
        f.write(f"R2 Score: {r2_score(y_test, lr_preds):.4f}\n\n")
        
        f.write("--- Random Forest Regressor ---\n")
        f.write(f"MSE: {mean_squared_error(y_test, rf_preds):.4f}\n")
        f.write(f"R2 Score: {r2_score(y_test, rf_preds):.4f}\n")

    # 9. Create a prediction function
    def predict_quality_of_sleep(sleep_duration, stress_level, exercise_frequency):
        input_data = pd.DataFrame(columns=features)
        input_data.loc[0] = [np.nan] * len(features)
        
        input_data['Sleep Duration'] = sleep_duration
        input_data['Stress Level'] = stress_level
        input_data['Exercise frequency'] = exercise_frequency
        
        input_imputed = pd.DataFrame(imputer.transform(input_data), columns=features)
        input_scaled = scaler.transform(input_imputed)
        
        prediction = rf_model.predict(input_scaled)
        return prediction[0]

    # Example Input Prediction
    example_prediction = predict_quality_of_sleep(6.5, 7, 2)
    
    # Append prediction to metrics
    print("--- Prediction Example ---")
    print(f"Inputs: Sleep Duration = 6.5, Stress Level = 7, Exercise frequency = 2")
    print(f"Predicted Quality of Sleep: {example_prediction:.2f}")

    with open('metrics_output.txt', 'a') as f:
        f.write("\n--- Prediction Example ---\n")
        f.write(f"Inputs: Sleep Duration = 6.5, Stress Level = 7, Exercise frequency = 2\n")
        f.write(f"Predicted Quality of Sleep: {example_prediction:.2f}\n")

    # 10. Feature Importance Plot
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [features[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main()
