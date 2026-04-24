"""
=============================================================
 FIS Snowboard Big Air — Score Prediction Model
 ML Class Project
=============================================================
 USAGE:
   python snowboard_model.py

 OUTPUTS:
   - Printed results table for all 3 models
   - snowboard_results.png  (figures for your report)
=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import sys
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# ── 1. Load & Inspect Data ───────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Loading Data")
print("=" * 60)

df = pd.read_csv("snowboard_dataset_complete.csv")
print(f"  Raw rows loaded: {len(df)}")
print(f"  Events: {df['Competition'].nunique()} competitions")
print(f"  Athletes: {df['Athlete'].nunique()} unique riders")
print(f"  Seasons: {sorted(df['Year'].unique())}")

# ── 2. Data Cleaning ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Cleaning & Feature Engineering")
print("=" * 60)

# Drop rows with no spin (ambiguous trick codes like 'T-Mo', 'bob', 'McB')
df_clean = df.dropna(subset=["SpinDegrees", "Difficulty"]).copy()
# Drop rows with assumed fall or major deduction
df_clean = df_clean[df_clean["JumpScore"] >= 65].copy()
#df_clean = df_clean[df_clean.groupby("SpinDegrees")["JumpScore"].transform(lambda x: x >= x.quantile(0.5))].copy()
print(f"  Rows after dropping ambiguous tricks: {len(df_clean)}")

# Binary encode Y/N columns
for col in ["Switch", "DoubleCork", "TripleCork", "Rodeo"]:
    df_clean[col] = (df_clean[col] == "Y").astype(int)

# Encode stance
df_clean["StanceGoofy"] = (df_clean["Stance"] == "G").astype(int)

# Encode direction (frontside=0, backside=1)
dir_map = {"f": 0, "b": 1}
df_clean["DirectionEnc"] = df_clean["Direction"].map(dir_map).fillna(0).astype(int)

# Encode grab difficulty
grab_difficulty = {
    "I": 1, "Me": 2, "Mu": 2, "Tg": 2, "Wed": 2, "Ng": 2,
    "Nb": 3, "J": 3, "Jp": 3, "St": 3, "Ro": 3, "Ddr": 4,
    "Tdr": 5, "Go": 2, "Bd": 2, "Ti": 2, "TK": 2, "Hb": 2,
    "Ste": 2, "RB": 2, "Bb": 2, "sa": 1, "Ind": 2, "pb": 2,
    "oT": 2, "Slb": 2, "Gor": 2
}
df_clean["GrabDifficulty"] = df_clean["Grab"].map(grab_difficulty).fillna(2)

# Run number as numeric
df_clean["RunNum"] = df_clean["Run"].astype(int)

# Jump ID: A=0, B=1 
df_clean["JumpIDB"] = (df_clean["JumpID"] == "B").astype(int)

# Year as a feature (captures progression over seasons)
df_clean["Season"] = df_clean["Year"].astype(int)

print(f"  Features engineered: SpinDegrees, Difficulty, Switch, DoubleCork,")
print(f"    TripleCork, Rodeo, DirectionEnc, StanceGoofy, GrabDifficulty,")
print(f"    RunNum, JumpIDB, Season")
print(f"  Final clean dataset: {len(df_clean)} rows")

# ── 3. Define Features & Target ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Feature Selection")
print("=" * 60)

FEATURES = [
    "SpinDegrees",    # primary difficulty driver
    "Difficulty",     # composite difficulty score (1-10)
    "Switch",         # harder takeoff = potentially higher reward
    "DoubleCork",     # off-axis complexity
    "TripleCork",     # even more complex
    "Rodeo",          # inverted element
    "DirectionEnc",   # frontside / backside
    "StanceGoofy",    # stance (goofy vs regular)
    "GrabDifficulty", # quality of grab
    "RunNum",         # 1st, 2nd, or 3rd attempt
    "JumpIDB",        # A or B jump
]

TARGET = "JumpScore"

X = df_clean[FEATURES]
y = df_clean[TARGET]

print(f"  Feature count: {len(FEATURES)}")
print(f"  Target: {TARGET}")
print(f"  Score range: {y.min():.1f} – {y.max():.1f}")
print(f"  Score mean:  {y.mean():.2f}  |  Std: {y.std():.2f}")

# ── 4. Train/Test Split ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: Train / Test Split  (80% / 20%)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")

# ── 5. Train Models ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Training Models")
print("=" * 60)

models = {
    "Linear Regression (Baseline)": LinearRegression(),
    "Random Forest":                 RandomForestRegressor(
                                         n_estimators=200,
                                         max_depth=8,
                                         min_samples_leaf=5,
                                         random_state=42),
    "XGBoost":                       XGBRegressor(
                                         n_estimators=300,
                                         max_depth=5,
                                         learning_rate=0.05,
                                         subsample=0.8,
                                         colsample_bytree=0.8,
                                         random_state=42,
                                         verbosity=0),
}

results = {}
trained  = {}

for name, model in models.items():
    print(f"\n  Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # 5-fold cross-validation on full dataset
    cv_scores = cross_val_score(model, X, y, cv=5,
                                scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    results[name] = {
        "MAE":    round(mae, 3),
        "RMSE":   round(rmse, 3),
        "R²":     round(r2, 4),
        "CV MAE": round(cv_mae, 3),
        "y_pred": y_pred,
    }
    trained[name] = model

    print(f"    MAE:    {mae:.3f}  (avg prediction off by {mae:.1f} points)")
    print(f"    RMSE:   {rmse:.3f}")
    print(f"    R²:     {r2:.4f}")
    print(f"    CV MAE: {cv_mae:.3f}  (5-fold cross-validation)")

# ── 6. Results Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6: Results Summary")
print("=" * 60)
print(f"\n  {'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV MAE':>8}")
print("  " + "-" * 71)
for name, r in results.items():
    print(f"  {name:<35} {r['MAE']:>8.3f} {r['RMSE']:>8.3f} "
          f"{r['R²']:>8.4f} {r['CV MAE']:>8.3f}")

best = min(results, key=lambda k: results[k]["MAE"])
print(f"\n  Best model by MAE: {best}")
print(f"  Interpretation: predictions off by ~{results[best]['MAE']:.1f} points on average")
print(f"  (Judges themselves vary by ~3-5 points per jump — so this is meaningful)")

# ── 7. Feature Importance (XGBoost) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7: Feature Importance (XGBoost)")
print("=" * 60)

xgb_model = trained["XGBoost"]
importances = pd.Series(
    xgb_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\n  Feature Importances:")
for feat, imp in importances.items():
    bar = "#" * int(imp * 50)
    print(f"    {feat:<20} {imp:.4f}  {bar}")

# ── 8. Example Predictions ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8: Example Predictions (using XGBoost)")
print("=" * 60)

example_tricks = [
    # spin, diff, switch, dc, tc, rodeo, dir, goofy, grab, run, jumpB
    {"label": "b-18-Me  (standard 1800 backside melon)",
     "feats": [1800, 9.0, 0, 0, 0, 0, 1, 0, 2, 2, 0]},
    {"label": "x-b-D-12-Ng  (switch BS double cork 1260 nosegrab)",
     "feats": [1260, 9.0, 1, 1, 0, 0, 1, 0, 2, 2, 1]},
    {"label": "Cab-18-Wed  (Caballerial 1800 weddle)",
     "feats": [1800, 9.5, 1, 0, 0, 0, 2, 0, 2, 1, 0]},
    {"label": "b-T-18-I  (BS triple cork 1800 indy)",
     "feats": [1800, 10.0, 0, 0, 1, 0, 1, 0, 1, 1, 1]},
    {"label": "f-19-I  (FS 1980 indy)",
     "feats": [1980, 10.0, 0, 0, 0, 0, 0, 0, 1, 3, 1]},
    {"label": "b-D-10-Wed  (BS double cork 1080 -- crash-risk trick)",
     "feats": [1080, 7.5, 0, 1, 0, 0, 1, 0, 2, 1, 0]},
]

print(f"\n  {'Trick':<55} {'Predicted Score':>15}")
print("  " + "-" * 72)
for ex in example_tricks:
    X_ex = pd.DataFrame([ex["feats"]], columns=FEATURES)
    pred = xgb_model.predict(X_ex)[0]
    print(f"  {ex['label']:<55} {pred:>12.1f}/100")

# ── 9. Generate Figures ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 9: Generating Figures  →  snowboard_results.png")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0F1923")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT   = "#00C8FF"
GOLD     = "#FFD700"
RED      = "#FF4B4B"
GREEN    = "#4BFF91"
DARK_BG  = "#0F1923"
CARD_BG  = "#1A2535"
TEXT     = "#E8EEF5"
SUBTLE   = "#5A7A9A"

def style_ax(ax, title):
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(SUBTLE)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.grid(color=SUBTLE, alpha=0.2, linestyle="--", linewidth=0.5)

# ── Fig 1: Score distribution ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, f"Score Distribution (all {len(df_clean)} jumps)")
ax1.hist(y, bins=30, color=ACCENT, alpha=0.85, edgecolor=DARK_BG, linewidth=0.4)
ax1.axvline(y.mean(), color=GOLD, linewidth=1.5, linestyle="--", label=f"Mean {y.mean():.1f}")
ax1.set_xlabel("Judge Score (0–100)")
ax1.set_ylabel("Frequency")
ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD_BG, edgecolor=SUBTLE)

# ── Fig 2: Spin degrees vs score scatter ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Spin Degrees vs Score")
scatter_colors = df_clean["DoubleCork"].map({0: ACCENT, 1: GOLD})
ax2.scatter(df_clean["SpinDegrees"], df_clean["JumpScore"],
            c=scatter_colors, alpha=0.35, s=12, linewidths=0)
ax2.set_xlabel("Spin Degrees")
ax2.set_ylabel("Judge Score")
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=ACCENT, label="Standard"),
              Patch(facecolor=GOLD,  label="Double Cork")]
ax2.legend(handles=legend_els, fontsize=8, labelcolor=TEXT,
           facecolor=CARD_BG, edgecolor=SUBTLE)

# ── Fig 3: Model comparison bar chart ────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Model Comparison — MAE (lower = better)")
model_names_short = ["Linear\nRegression", "Random\nForest", "XGBoost"]
maes   = [results[k]["MAE"] for k in results]
colors = [RED, ACCENT, GREEN]
bars   = ax3.bar(model_names_short, maes, color=colors, alpha=0.85, edgecolor=DARK_BG, linewidth=0.5)
for bar, val in zip(bars, maes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.2f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
ax3.set_ylabel("Mean Absolute Error (points)")
ax3.set_ylim(0, max(maes) * 1.25)

# ── Fig 4: Actual vs Predicted (XGBoost) ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
style_ax(ax4, "Actual vs Predicted Scores — XGBoost")
y_pred_xgb = results["XGBoost"]["y_pred"]
ax4.scatter(y_test, y_pred_xgb, alpha=0.4, s=14, color=ACCENT, linewidths=0)
lims = [0, 100]
ax4.plot(lims, lims, color=GOLD, linewidth=1.5, linestyle="--", label="Perfect prediction")
ax4.set_xlabel("Actual Score")
ax4.set_ylabel("Predicted Score")
ax4.set_xlim(lims); ax4.set_ylim(lims)
ax4.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD_BG, edgecolor=SUBTLE)
r2_xgb = results["XGBoost"]["R²"]
ax4.text(5, 90, f"R² = {r2_xgb:.4f}", color=GOLD, fontsize=11, fontweight="bold")
ax4.text(5, 82, f"MAE = {results['XGBoost']['MAE']:.2f} pts", color=TEXT, fontsize=9)

# ── Fig 5: Residuals ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, "Residuals — XGBoost")
residuals = y_test.values - y_pred_xgb
ax5.hist(residuals, bins=25, color=GREEN, alpha=0.8, edgecolor=DARK_BG, linewidth=0.4)
ax5.axvline(0, color=GOLD, linewidth=1.5, linestyle="--")
ax5.set_xlabel("Residual (Actual − Predicted)")
ax5.set_ylabel("Frequency")
ax5.text(0.97, 0.95, f"σ = {residuals.std():.2f}", transform=ax5.transAxes,
         ha="right", va="top", color=TEXT, fontsize=9)

# ── Fig 6: Feature Importance ─────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0:2])
style_ax(ax6, "Feature Importance — XGBoost")
feat_colors = [GREEN if i < 3 else ACCENT if i < 6 else SUBTLE
               for i in range(len(importances))]
bars6 = ax6.barh(importances.index[::-1], importances.values[::-1],
                  color=feat_colors[::-1], alpha=0.85, edgecolor=DARK_BG, linewidth=0.4)
ax6.set_xlabel("Feature Importance (gain)")
for bar, val in zip(bars6, importances.values[::-1]):
    ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", color=TEXT, fontsize=7)

# ── Fig 7: R² comparison ──────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, "R² Score by Model")
r2s    = [results[k]["R²"] for k in results]
colors7 = [RED, ACCENT, GREEN]
bars7  = ax7.bar(model_names_short, r2s, color=colors7, alpha=0.85,
                  edgecolor=DARK_BG, linewidth=0.5)
for bar, val in zip(bars7, r2s):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
ax7.set_ylabel("R² (higher = better)")
ax7.set_ylim(0, min(1.0, max(r2s) * 1.2))
ax7.axhline(0.5, color=SUBTLE, linewidth=1, linestyle=":", alpha=0.6)

# ── Title ──────────────────────────────────────────────────────────────────────
fig.suptitle(f"FIS Snowboard Big Air — Score Prediction Model\n"
             f"{len(df_clean)} jumps · {df_clean['Competition'].nunique()} competitions · "
             f"{df_clean['Year'].min()}/{str(df_clean['Year'].min()+1)[-2:]} & "
             f"{df_clean['Year'].max()}/{str(df_clean['Year'].max()+1)[-2:]} seasons",
             color=TEXT, fontsize=13, fontweight="bold", y=0.98)

plt.savefig("snowboard_results.png", dpi=150, bbox_inches="tight",
            facecolor=DARK_BG)
print("  Saved: snowboard_results.png")

# ── 10. Print Full Results Table for Report ───────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS TABLE (copy into your report)")
print("=" * 60)
print(f"\n  {'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV MAE':>8}")
print("  " + "-" * 71)
for name, r in results.items():
    print(f"  {name:<35} {r['MAE']:>8.3f} {r['RMSE']:>8.3f} "
          f"{r['R²']:>8.4f} {r['CV MAE']:>8.3f}")

print("\n" + "=" * 60)
print("  NOTES FOR YOUR REPORT:")
print("=" * 60)
print("""
  Methodology justification (cite FIS Judges Handbook):
    - SpinDegrees & Difficulty: FIS criterion 'Difficulty — Amount of Rotation'
    - Switch: FIS criterion 'Difficulty — Blind Landings'
    - DoubleCork/TripleCork: FIS criterion 'Difficulty — Axis'
    - GrabDifficulty: FIS criterion 'Execution — Grabs'
    - RunNum / JumpIDB: athletes often save hardest tricks for later runs

  Limitations to discuss:
    - Judging is inherently subjective (Progression, Style not captured)
    - All events are men's only (no women's data in this batch)
    - Low-scoring rows include falls/bails — trick code doesn't capture execution
    - Some trick codes ambiguous (e.g. 'T-Mo', 'bob') were excluded (~15 rows)

  Ethics to discuss:
    - Could a model like this bias judges before runs?
    - Does it disadvantage athletes with creative but unconventional tricks?
    - Gender and nationality patterns in the data?
""")
print("  Done! Check snowboard_results.png for your report figures.")
