import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Loading and preprocessing datasets...")
    # 1. Load datasets
    df1 = pd.read_csv('Sleep_Efficiency.csv')
    df2 = pd.read_csv('data111.csv')

    # Rename variable to synchronize formats
    if 'Sleep duration' in df1.columns:
        df1 = df1.rename(columns={'Sleep duration': 'Sleep Duration'})

    # Map target functionally
    if 'Sleep efficiency' in df1.columns:
        df1['Quality of Sleep'] = df1['Sleep efficiency'] * 10
    
    # Merge datasets correctly
    df = pd.concat([df1, df2], ignore_index=True)

    # Convert categorical attributes
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Smoking status']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes.replace(-1, np.nan)

    # Set explicit modeling features explicitly requested by user
    features = ['Sleep Duration', 'Sleep efficiency', 'Stress Level', 
                'Physical Activity Level', 'Heart Rate', 'Daily Steps', 
                'Awakenings', 'Exercise frequency', 'Gender', 'Occupation', 'BMI Category', 'Smoking status']
    target = 'Quality of Sleep'

    all_cols = features + [target]
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_subset = df[all_cols].copy()

    # Drop target empty rows just in case any completely invalid rows exist
    df_subset = df_subset.dropna(subset=[target])

    # Display original features
    original_features = features.copy()
    
    print("Imputing variables...")
    # 2. Imputing NaN instances properly
    imputer = SimpleImputer(strategy='median')
    df_subset[features] = imputer.fit_transform(df_subset[features])

    # 3. EDA - Correlation Heatmap
    print("Generating EDA visualizations...")
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_subset.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Base splits
    X = df_subset[features]
    y = df_subset[target]

    # Normalize elements
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Automatic Feature Selection
    print("Performing feature selection using RandomForest weights...")
    base_selector = RandomForestRegressor(n_estimators=50, random_state=42)
    base_selector.fit(X_scaled, y)
    
    # Store raw importances
    base_importances = base_selector.feature_importances_
    
    # Use SelectFromModel to keep only meaningful features
    selector = SelectFromModel(base_selector, prefit=True, threshold='0.5*median')
    # Filter features dynamically
    feature_mask = selector.get_support()
    selected_features = [features[i] for i, mask in enumerate(feature_mask) if mask]
    
    print(f"Features selected for pipeline ({len(selected_features)}/{len(features)}): {selected_features}")
    
    X_selected = X_scaled[:, feature_mask]

    # Split filtered features
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # 5. Pipeline - Train Multiple Models & Hyperparameters Tuning Iteratively
    print("Training models internally (Cross-validation)...")

    # A) Linear Regression (Base Validation)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_preds)

    # B) Random Forest Grids
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_preds = rf_best.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)

    # C) Gradient Boosting Grids
    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    gb_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    gb_search.fit(X_train, y_train)
    gb_best = gb_search.best_estimator_
    gb_preds = gb_best.predict(X_test)
    gb_r2 = r2_score(y_test, gb_preds)

    # 6. Assess Absolute Top
    models = {
        "Linear Regression": (lr, lr_preds, lr_r2),
        "Random Forest Regressor": (rf_best, rf_preds, rf_r2),
        "Gradient Boosting Regressor": (gb_best, gb_preds, gb_r2)
    }

    best_model_name = max(models, key=lambda k: models[k][2])
    best_model, best_preds, best_r2 = models[best_model_name]
    best_mse = mean_squared_error(y_test, best_preds)

    print("\n--- Model Evaluation ---")
    out_lines = ["--- Model Evaluation ---\n"]
    for name, (mod, preds, local_r2) in list(models.items()):
        local_mse = mean_squared_error(y_test, preds)
        rep = f"{name} -> MSE: {local_mse:.4f} | R2: {local_r2:.4f}"
        print(rep)
        out_lines.append(rep + "\n")

    print(f"\n>> BEST MODEL: {best_model_name} (R2={best_r2:.4f}) <<")
    out_lines.append(f"\n>> BEST MODEL: {best_model_name} (R2={best_r2:.4f}) <<\n")

    # 7. Visualization - Prediction vs Actual
    print("Generating Prediction vs Actual plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_preds, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Quality of Sleep')
    plt.ylabel('Predicted Quality of Sleep')
    plt.title(f'Prediction vs Actual ({best_model_name})')
    plt.tight_layout()
    plt.savefig('prediction_vs_actual.png')
    plt.close()

    # 8. Visualization - Feature Importance (only if forest or gradient boosting)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        # Sort and take top limits dynamically
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:len(selected_features)]
        
        top_feats = [selected_features[i] for i in top_indices]
        top_importances = importances[top_indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_feats, palette='viridis')
        plt.title(f'Feature Importances ({best_model_name})')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    # 9. Dump tools
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump({'features': original_features, 'mask': feature_mask}, 'selector.pkl')

    # Export to text directly
    with open('metrics_output.txt', 'w') as f:
        f.writelines(out_lines)

    # 10. Unified Reusable prediction closure function
    def predict_quality_of_sleep(sleep_duration, stress_level, exercise_frequency, heart_rate, daily_steps):
        # Create full dataframe of NaN default values matching original feature layout length
        input_data = pd.DataFrame(columns=original_features)
        input_data.loc[0] = [np.nan] * len(original_features)
        
        # Override specific assignments
        input_data['Sleep Duration'] = sleep_duration
        input_data['Stress Level'] = stress_level
        input_data['Exercise frequency'] = exercise_frequency
        input_data['Heart Rate'] = heart_rate
        input_data['Daily Steps'] = daily_steps
        
        input_imputed = pd.DataFrame(imputer.transform(input_data), columns=original_features)
        
        # Scale correctly against primary dimensions
        input_scaled = scaler.transform(input_imputed)
        
        # Apply strict subset threshold masking equivalent to selector filtering output sizes
        input_selected = input_scaled[:, feature_mask]
        
        prediction = best_model.predict(input_selected)
        return prediction[0]

    # Predict required constraint explicitly requested
    val = predict_quality_of_sleep(6.5, 7, 2, 75, 5000)
    
    with open('metrics_output.txt', 'a') as f:
        f.write("\n--- Test Prediction Execution ---\n")
        f.write(f"Inputs: Sleep Duration=6.5, Stress Level=7, Exercise frequency=2, Heart Rate=75, Daily Steps=5000\n")
        f.write(f"Predicted Quality of Sleep: {val:.3f}\n")
    print("\n--- Test Prediction Execution ---")
    print(f"Predicted Quality of Sleep: {val:.3f}")

if __name__ == "__main__":
    main()
