import pandas as pd
import numpy as np
import os
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_confusion_matrix_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    selection_rate = (tp + fp) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'tpr': tpr, 'fpr': fpr, 'selection_rate': selection_rate
    }

def calculate_fairness_metrics(df, predictions, sensitive_attr, target_label, privileged_value, unprivileged_value):
    """
    Calculate fairness metrics using confusion matrices for privileged and unprivileged groups.
    """
    df['prediction'] = (predictions > 0.5).astype(int)
    
    unprivileged_mask = df[sensitive_attr] == unprivileged_value
    privileged_mask = df[sensitive_attr] == privileged_value
    
    metrics_unprivileged = get_confusion_matrix_metrics(
        df[unprivileged_mask][target_label], 
        df[unprivileged_mask]['prediction']
    )
    
    metrics_privileged = get_confusion_matrix_metrics(
        df[privileged_mask][target_label], 
        df[privileged_mask]['prediction']
    )
    
    spd = metrics_unprivileged['selection_rate'] - metrics_privileged['selection_rate']
    eod = metrics_unprivileged['tpr'] - metrics_privileged['tpr']
    aod = 0.5 * ((metrics_unprivileged['fpr'] - metrics_privileged['fpr']) + 
                 (metrics_unprivileged['tpr'] - metrics_privileged['tpr']))
    
    return {
        'spd': spd,
        'eod': eod,
        'aod': aod,
        'unprivileged': metrics_unprivileged,
        'privileged': metrics_privileged
    }

def plot_fairness_results(results, sensitive_attr):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Confusion Matrix Unprivileged
    m_un = results['unprivileged']
    cm_un = np.array([[m_un['tn'], m_un['fp']], [m_un['fn'], m_un['tp']]])
    im1 = axes[0].imshow(cm_un, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title(f'Confusion Matrix: Unprivileged ({sensitive_attr}=0)')
    fig.colorbar(im1, ax=axes[0])
    
    # 2. Confusion Matrix Privileged
    m_pr = results['privileged']
    cm_pr = np.array([[m_pr['tn'], m_pr['fp']], [m_pr['fn'], m_pr['tp']]])
    im2 = axes[1].imshow(cm_pr, interpolation='nearest', cmap=plt.cm.Oranges)
    axes[1].set_title(f'Confusion Matrix: Privileged ({sensitive_attr}=1)')
    fig.colorbar(im2, ax=axes[1])
    
    for ax, cm in zip([axes[0], axes[1]], [cm_un, cm_pr]):
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    # 3. Fairness Metrics Bar Chart
    metrics = ['SPD', 'EOD', 'AOD']
    values = [results['spd'], results['eod'], results['aod']]
    axes[2].bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_title('Fairness Metrics Comparison')
    for i, v in enumerate(values):
        axes[2].text(i, v + (0.05 if v >= 0 else -0.1), f'{v:.4f}', ha='center')

    plt.tight_layout()

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
    
    fairness_results = calculate_fairness_metrics(df, predictions, sensitive_attr, 'Class-label', privileged_value, unprivileged_value)
    
    print(f"\nGroup Fairness Metrics for attribute: {sensitive_attr}")
    print(f"Statistical Parity Difference (SPD): {fairness_results['spd']:.4f}")
    print(f"Equal Opportunity Difference (EOD): {fairness_results['eod']:.4f}")
    print(f"Average Odds Difference (AOD): {fairness_results['aod']:.4f}")
    
    # Plot results
    print("Generating visualization...")
    plot_fairness_results(fairness_results, sensitive_attr)
    
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
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
