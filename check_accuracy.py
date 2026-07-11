import sys
sys.path.insert(0, '/Users/mohamedalaa/Downloads/NHA-4-190')

from src.utils import load_config, load_artifact
from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.model_training import split_data
from src.model_evaluation import evaluate_model, find_best_threshold
from sklearn.metrics import accuracy_score

config = load_config("config/config.yaml")
df_clean = run_preprocessing("config/config.yaml")
df_eng = run_feature_engineering(df_clean, config)
X_train, X_train_smote, X_test, y_train, y_train_smote, y_test = split_data(df_eng, config)

artifact_cfg = config["artifacts"]
model_dir = artifact_cfg["model_dir"]

models = {
    "Random Forest (Baseline)":  load_artifact(f"{model_dir}/random_forest_model.joblib"),
    "Random Forest (Optimized)": load_artifact(f"{model_dir}/best_model.joblib"),
    "XGBoost (Baseline)":        load_artifact(f"{model_dir}/xgboost_model.joblib"),
}

print("="*60)
print("  MODEL EVALUATION RESULTS")
print("="*60)

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    
    # Get full report for precision/recall/f1/roc-auc
    result = evaluate_model(model, X_test, y_test, model_name=name, save_plot=False)
    
    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
    print("-" * 40)
    print(result['report'])
    
    # Best threshold
    thresh = find_best_threshold(model, X_test, y_test, model_name=name)
    print(f"  Best Threshold (F2): {thresh:.3f}")
    print("="*60)
