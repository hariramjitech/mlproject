import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_data():
    """Load Breast Cancer dataset"""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names

def evaluate_model_cv(model, X, y, cv, model_name):
    """Evaluate model using Cross-Validation"""
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    results = {}
    
    print(f"Evaluating {model_name} with {cv.get_n_splits()}-fold CV...")
    
    for score_name in scoring:
        scores = cross_val_score(model, X, y, cv=cv, scoring=score_name)
        results[score_name] = np.mean(scores)
        print(f"  {score_name}: {np.mean(scores):.4f}")
        
    return results

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

def main():
    # 1. Load Data
    X, y, feature_names, target_names = load_data()
    
    # Scale data for SVM (and good practice for others)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Define Cross-Validation
    # Using Stratified K-Fold to maintain class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Define Models to Compare
    # Base Random Forest (before tuning)
    rf_base = RandomForestClassifier(random_state=42)
    
    # SVM
    svm = SVC(kernel='linear', random_state=42)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)

    results = {}

    # 4. Evaluate Base Models
    models = {
        'Random Forest (Base)': rf_base,
        'SVM': svm,
        'Decision Tree': dt
    }

    for name, model in models.items():
        results[name] = evaluate_model_cv(model, X_scaled, y, cv, name)

    # 5. Hyperparameter Tuning for Random Forest
    print("\nTuning Random Forest Hyperparameters...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3, # 3-fold for tuning to save time
                           n_jobs=-1,
                           verbose=1)
    
    rf_grid.fit(X_scaled, y)
    best_rf = rf_grid.best_estimator_
    print(f"Best Parameters: {rf_grid.best_params_}")
    
    # Evaluate Tuned Random Forest
    results['Random Forest (Tuned)'] = evaluate_model_cv(best_rf, X_scaled, y, cv, 'Random Forest (Tuned)')

    # 6. Save Metrics
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # 7. Generate Comparison Plots
    
    # Accuracy Comparison
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies, palette='viridis')
    plt.title('Model Comparison - Cross-Validation Accuracy')
    plt.ylim(0.8, 1.0) # Zoom in to show differences
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_accuracy_comparison.png')
    plt.close()
    
    # Support details for detailed plotting (using a simple train/test split for confusion matrix viz)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    # Fit and plot confusion matrix for Best RF
    best_rf.fit(X_train, y_train)
    y_pred_rf = best_rf.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_rf, target_names, 'Random Forest (Tuned) Confusion Matrix', 'rf_tuned_cm.png')
    
    # Fit and plot for SVM
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_svm, target_names, 'SVM Confusion Matrix', 'svm_cm.png')

    print("\nEvaluation complete. Results saved.")

if __name__ == "__main__":
    main()
