import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import scipy.stats as stats

# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================
df = pd.read_csv("data/csfloat_butterfly_knife.csv")
print(f"Wczytano: {len(df):,} obserwacji")

# ============================================================
# 2. CZYSZCZENIE DANYCH
# ============================================================

# Usunięcie Vanilla (float_value == 0)
n_przed = len(df)
df = df[df["float_value"] > 0].copy()
print(f"Usunięto Vanilla (float=0):  {n_przed - len(df)} obserwacji")

# Usunięcie outlierów > $30,000
n_przed = len(df)
df = df[df["price_usd"] <= 30000].copy()
print(f"Usunięto outlier > $30k:     {n_przed - len(df)} obserwacji")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Nazwa skina
df["skin_name"] = (
    df["market_hash_name"]
    .str.replace(r"★ (StatTrak™ )?Butterfly Knife \| ", "", regex=True)
    .str.replace(r"\s*\(.*?\)", "", regex=True)
    .str.strip()
    .replace("", "Vanilla")
    .fillna("Vanilla")
)

# Wear z nazwy
df["wear"] = (
    df["market_hash_name"]
    .str.extract(r"\((.*?)\)$")[0]
    .fillna("Vanilla")
)

# Usunięcie Vanilla z floatem
n_przed = len(df)
df = df[df["wear"] != "Vanilla"].copy()
print(f"Usunięto Vanilla z floatem:  {n_przed - len(df)} obserwacji")
print(f"Finalny zbiór:               {len(df):,} obserwacji")

# Wear porządkowy
wear_order = {
    "Factory New":    1,
    "Minimal Wear":   2,
    "Field-Tested":   3,
    "Well-Worn":      4,
    "Battle-Scarred": 5
}
df["wear_num"] = df["wear"].map(wear_order)

# StatTrak 0/1
df["stattrak"] = df["is_stattrak"].astype(int)

# Log-transformacje
df["log_price"] = np.log(df["price_usd"])
df["log_float"] = np.log(df["float_value"])

# Faza Dopplera przez paint_index
doppler_phase_map = {
    418: "Phase_1",   618: "Phase_2",
    420: "Phase_3",   421: "Phase_4",
    415: "Ruby",      619: "Sapphire",
    617: "Black_Pearl", 568: "Emerald",
    569: "GD_Phase_1", 570: "GD_Phase_2",
    571: "GD_Phase_3", 572: "GD_Phase_4",
}
df["doppler_phase"] = df["paint_index"].map(doppler_phase_map)

# Flaga is_gem
gem_indexes = [415, 619, 617, 568]
df["is_gem"] = (
    df["market_hash_name"].str.contains("Doppler") &
    df["paint_index"].isin(gem_indexes)
).astype(int)

# Flaga is_blue_gem
blue_gem_seeds = [
    182, 494, 194, 283, 330, 694, 703, 838, 891, 913, 917,
    29, 214, 263, 371, 396, 398, 478, 488, 510, 612, 638,
    652, 717, 749, 800, 868, 881, 928
]
df["is_blue_gem"] = (
    df["skin_name"].str.contains("Case Hardened", na=False) &
    df["paint_seed"].isin(blue_gem_seeds)
).astype(int)

# Dummy variables — baza: Boreal Forest (najtańszy skin)
skin_dummies = pd.get_dummies(df["skin_name"], prefix="skin").astype(int)
skin_dummies = skin_dummies.drop(columns=["skin_Boreal Forest"])
df = pd.concat([df, skin_dummies], axis=1)

print(f"\nFeature engineering — podsumowanie:")
print(f"  Dummy skinów:   {skin_dummies.shape[1]}")
print(f"  Gemów:          {df['is_gem'].sum()}")
print(f"  Blue Gemów:     {df['is_blue_gem'].sum()}")
print(f"  StatTrak:       {df['stattrak'].sum()}")

# ============================================================
# 4. BUDOWA MODELU OLS LOG-LOG
# ============================================================

# Jawne wybranie kolumn numerycznych — wykluczamy teksty
skin_dummy_cols = [c for c in skin_dummies.columns]

feature_cols = (
    ["log_float", "wear_num", "stattrak", "is_gem", "is_blue_gem"] +
    skin_dummy_cols
)

X = df[feature_cols].astype(np.float64)
y = df["log_price"].astype(np.float64)

print(f"\nMacierz X: {X.shape} | NaN: {X.isnull().sum().sum()}")

X_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_const).fit()
print(model_ols.summary())

# ============================================================
# 5. METRYKI NA ORYGINALNEJ SKALI
# ============================================================
y_pred = np.exp(model_ols.fittedvalues)
y_true = df["price_usd"]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)

print(f"\n{'='*45}")
print(f"METRYKI OLS — ORYGINALNA SKALA (USD)")
print(f"{'='*45}")
print(f"R²:   {model_ols.rsquared:.4f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE:  ${mae:,.2f}")

# ============================================================
# 6. INTERPRETACJA WSPÓŁCZYNNIKÓW
# ============================================================
print(f"\n{'='*45}")
print(f"INTERPRETACJA WSPÓŁCZYNNIKÓW")
print(f"(baza porównania: Boreal Forest)")
print(f"{'='*45}")

key_vars = [
    "log_float", "wear_num", "stattrak",
    "is_gem", "is_blue_gem",
    "skin_Doppler", "skin_Fade", "skin_Slaughter",
    "skin_Gamma Doppler", "skin_Marble Fade"
]

for var in key_vars:
    if var in model_ols.params:
        coef   = model_ols.params[var]
        pval   = model_ols.pvalues[var]
        effect = (np.exp(coef) - 1) * 100
        sig    = "✅" if pval < 0.05 else "❌ nieistotna"
        print(f"{var:<25} {effect:+.1f}%   p={pval:.3f} {sig}")

# ============================================================
# 7. DIAGNOSTYKA RESZT
# ============================================================
residuals = model_ols.resid
fitted    = model_ols.fittedvalues

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("OLS Log-Log — Diagnostyka reszt", fontsize=14, fontweight="bold")

axes[0].scatter(fitted, residuals, alpha=0.3, s=8, color="steelblue")
axes[0].axhline(0, color="red", linewidth=1.5)
axes[0].set_xlabel("Wartości dopasowane (log_price)")
axes[0].set_ylabel("Reszty")
axes[0].set_title("Reszty vs Wartości dopasowane")

stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot reszt")

axes[2].hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
axes[2].axvline(0, color="red", linewidth=1.5)
axes[2].set_xlabel("Reszty")
axes[2].set_ylabel("Liczba obserwacji")
axes[2].set_title("Rozkład reszt")

plt.tight_layout()
plt.savefig("figures/ols_diagnostyka_reszt.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: ols_diagnostyka_reszt.png")

# ============================================================
# 8. ZAPIS FINALNEGO PLIKU Z FEATURAMI
# ============================================================
df.to_csv("data/csfloat_butterfly_knife_features.csv", index=False)
print("Zapisano: csfloat_butterfly_knife_features.csv")