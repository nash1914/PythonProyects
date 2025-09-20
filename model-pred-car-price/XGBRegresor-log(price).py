import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# === 1. Cargar dataset ===
df = pd.read_csv("vehicle_price_prediction.csv")

# === 2. Limpieza de accident_history ===
df["accident_history"] = df["accident_history"].replace(
    ["", " ", "NA", "N/A", "na", "null", None, np.nan],
    "None"
)

# === 3. Conversión de columnas numéricas con comas ===
if "brand_popularity" in df.columns:
    df["brand_popularity"] = (
        df["brand_popularity"].astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
if "price" in df.columns:
    df["price"] = (
        df["price"].astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# === 4. Separar X e y ===
X = df.drop("price", axis=1)
y = df["price"]

# === 5. Definir columnas ===
categorical_cols = ["make", "model", "transmission", "fuel_type",
                    "drivetrain", "body_type", "accident_history",
                    "seller_type", "condition", "trim"]

numeric_cols = ["year", "mileage", "engine_hp", "owner_count",
                "vehicle_age", "brand_popularity"]

# === 6. Preprocesamiento ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# === 7. Modelo base (sin log todavía) ===
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# === 8. Modelo con transformación log <-> exp ===
xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor=xgb,
        func=np.log1p,      # aplica log(1+y) antes de entrenar
        inverse_func=np.expm1  # aplica exp(y)-1 al predecir
    ))
])

print("Entrenando modelo con log(price) dentro del pipeline... ⏳")
xgb_model.fit(X, y)

# === 9. Guardar modelo entrenado ===
joblib.dump(xgb_model, "xgb_model_log_pipeline.pkl")

# === 10. Validación Cruzada ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = cross_val_score(xgb_model, X, y, cv=kf,
                             scoring=make_scorer(mean_squared_error))
r2_scores = cross_val_score(xgb_model, X, y, cv=kf,
                            scoring=make_scorer(r2_score))

print("Resultados Cross-Validation con log-transform automático:")
print(f"Promedio MSE: {np.mean(mse_scores):,.2f}")
print(f"Promedio R²: {np.mean(r2_scores):.4f}")

# === 11. Ejemplo de predicciones ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = xgb_model.predict(X_test[:5])  # ✅ ya devuelve precios en escala original
print("\n=== Ejemplo de Predicciones ===")
print("Predicciones:", np.round(y_pred, 2))
print("Valores reales:", y_test[:5].values)
