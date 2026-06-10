import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ── Load data 
#──────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('features_with_labels.csv')

print(f"Total road segments: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ── Drop NaN rows 
#──────────────────────────────────────────────
df = df.dropna()
print(f"After dropping NaN: {len(df)}")

# ── Features and target 
#────────────────────────────────────────
feature_cols = [
    'streetlight_score',
    'cctv_score',
    'police_proximity',
    'bus_stop_score'
]

X = df[feature_cols].values
y = df['safety_label'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"\nLabel distribution:")
print(pd.Series(y).describe())

# ── Train test split 
#───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ── Train XGBoost 
#──────────────────────────────────────────────
print("\nTraining model...")
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── Evaluate 
#───────────────────────────────────────────────────
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nR² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")

# ── Feature importance 
#─────────────────────────────────────────
print("\nFeature Importance:")
for feat, imp in zip(feature_cols, model.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# ── Save model 
#─────────────────────────────────────────────────
joblib.dump(model, 'safety_model.pkl')
print("\nModel saved to safety_model.pkl")
