import pandas as pd
import numpy as np
import os
from keras.models import load_model

def calculate_fairness_metrics(df, predictions, sensitive_attr, target_label, privileged_value, unprivileged_value):
    """
    Calculate basic fairness metrics: Statistical Parity Difference and Equal Opportunity Difference.
    """
    # Add predictions to the dataframe
    df['prediction'] = (predictions > 0.5).astype(int)
    
    # Statistical Parity Difference
    # P(Y_hat=1 | sensitive=unprivileged) - P(Y_hat=1 | sensitive=privileged)
    prob_unprivileged = df[df[sensitive_attr] == unprivileged_value]['prediction'].mean()
    prob_privileged = df[df[sensitive_attr] == privileged_value]['prediction'].mean()
    spd = prob_unprivileged - prob_privileged
    
    # Equal Opportunity Difference
    # P(Y_hat=1 | Y=1, sensitive=unprivileged) - P(Y_hat=1 | Y=1, sensitive=privileged)
    df_pos = df[df[target_label] == 1]
    prob_unprivileged_pos = df_pos[df_pos[sensitive_attr] == unprivileged_value]['prediction'].mean()
    prob_privileged_pos = df_pos[df_pos[sensitive_attr] == privileged_value]['prediction'].mean()
    eod = prob_unprivileged_pos - prob_privileged_pos
    
    return spd, eod

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load the dataset
    dataset_path = os.path.join(script_dir, 'dataset', 'processed_adult.csv')
    df = pd.read_csv(dataset_path)
    
    # 2. Prepare features and labels
    # The last column is the target label 'Class-label'
    X = df.drop('Class-label', axis=1).values
    y_true = df['Class-label'].values
    
    # 3. Load the pre-trained model
    model_path = os.path.join(script_dir, 'DNN', 'model_processed_adult.h5')
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # 4. Perform inference
    print("Performing inference...")
    predictions = model.predict(X)
    
    # 5. Calculate fairness metrics for 'gender'
    sensitive_attr = 'gender'
    privileged_value = 1
    unprivileged_value = 0
    
    spd, eod = calculate_fairness_metrics(df, predictions, sensitive_attr, 'Class-label', privileged_value, unprivileged_value)
    
    print(f"\nGroup Fairness Metrics for attribute: {sensitive_attr}")
    print(f"Statistical Parity Difference (SPD): {spd:.4f}")
    print(f"Equal Opportunity Difference (EOD): {eod:.4f}")
    
    # 6. Individual Fairness Test (Flipping sensitive attribute)
    print("\nPerforming Individual Fairness test (flipping gender)...")
    X_flipped = X.copy()
    gender_idx = df.columns.get_loc(sensitive_attr)
    # Flip gender: 1 -> 0, 0 -> 1
    X_flipped[:, gender_idx] = 1 - X_flipped[:, gender_idx]
    
    predictions_flipped = model.predict(X_flipped)
    predictions_binary = (predictions > 0.5).astype(int)
    predictions_flipped_binary = (predictions_flipped > 0.5).astype(int)
    
    num_flipped = np.sum(predictions_binary != predictions_flipped_binary)
    total = len(df)
    print(f"Number of individuals whose prediction changed after flipping gender: {num_flipped} / {total} ({100*num_flipped/total:.2f}%)")
    
    if num_flipped == 0:
        print("The model satisfies individual fairness for the gender attribute on this dataset.")
    else:
        print(f"The model exhibits individual unfairness for {num_flipped} instances.")

if __name__ == "__main__":
    main()
