import xgboost as xgb
print("XGBoost version:", xgb.__version__)


from xgboost import XGBClassifier

model = XGBClassifier(tree_method='gpu_hist')
print("Model created OK.")