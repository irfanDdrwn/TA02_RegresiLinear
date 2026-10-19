"""Melatih model Regresi Linear Sederhana: Berat Ikan (Weight) ~ Panjang Ikan (Length3).

Jalankan:  python train_model.py
Output  :  model/fish_model.joblib, model/metadata.json, static/img/*.png
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURE, TARGET = "Length3", "Weight"

# 1. Muat data
df = pd.read_csv(os.path.join(BASE, "Fish.csv"), encoding="utf-8-sig")
print("Bentuk data:", df.shape)
print(df.head())
print("Missing values:", int(df.isna().sum().sum()))

# 2. Pembersihan: berat 0 gram tidak masuk akal (kesalahan pencatatan)
n_awal = len(df)
df = df[df[TARGET] > 0].reset_index(drop=True)
print(f"Baris dihapus (berat <= 0): {n_awal - len(df)}")

os.makedirs(os.path.join(BASE, "static/img"), exist_ok=True)

# 3. EDA: korelasi dan scatterplot
corr = df[[TARGET, "Length1", "Length2", "Length3", "Height", "Width"]].corr()[TARGET]
print("\nKorelasi dengan Weight:\n", corr.round(3))

sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=FEATURE, y=TARGET, hue="Species")
plt.title("Panjang Ikan vs Berat Ikan")
plt.xlabel("Panjang ikan (cm)")
plt.ylabel("Berat ikan (gram)")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "static/img/scatter.png"), dpi=130)
plt.close()

# 4. Split data & latih model
X, y = df[[FEATURE]], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

# 5. Evaluasi
r2 = r2_score(y_test, pred)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = mean_absolute_error(y_test, pred)
print(f"\nIntercept : {model.intercept_:.4f}")
print(f"Koefisien : {model.coef_[0]:.4f}")
print(f"R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

# Grafik garis regresi
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, label="Data latih", alpha=0.7)
plt.scatter(X_test, y_test, label="Data uji", alpha=0.9, color="orange")
xs = np.linspace(X[FEATURE].min(), X[FEATURE].max(), 100).reshape(-1, 1)
plt.plot(xs, model.predict(pd.DataFrame(xs, columns=[FEATURE])), color="red", lw=2, label="Garis regresi")
plt.title("Garis Regresi Linear")
plt.xlabel("Panjang ikan (cm)")
plt.ylabel("Berat ikan (gram)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, "static/img/regression_line.png"), dpi=130)
plt.close()

# Grafik aktual vs prediksi
plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred)
lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lim, lim, color="red", lw=2)
plt.title("Berat Aktual vs Berat Prediksi (data uji)")
plt.xlabel("Berat aktual (gram)")
plt.ylabel("Berat prediksi (gram)")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "static/img/actual_vs_pred.png"), dpi=130)
plt.close()

# 6. Simpan model (Joblib) + metadata untuk aplikasi web
os.makedirs(os.path.join(BASE, "model"), exist_ok=True)
joblib.dump(model, os.path.join(BASE, "model/fish_model.joblib"))
meta = {
    "feature": FEATURE,
    "intercept": float(model.intercept_),
    "coef": float(model.coef_[0]),
    "r2": float(r2),
    "rmse": rmse,
    "mae": float(mae),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "min_length": float(X[FEATURE].min()),
    "max_length": float(X[FEATURE].max()),
    "zero_weight_length": float(-model.intercept_ / model.coef_[0]),
}
with open(os.path.join(BASE, "model/metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("\nModel dan metadata tersimpan.")
