"""
High-value order prediction for a sports-equipment wholesaler.
Run: python order_value_model.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
os.makedirs("figures", exist_ok=True)
SEED = 42

# =====================================================================
# 1. DATA COLLECTION AND PREPARATION
# =====================================================================
raw = pd.read_excel("data/sports_sales.xlsx")
print("Raw shape:", raw.shape)
print("Duplicate rows:", int(raw.duplicated().sum()), " Duplicate order numbers:", int(raw["Order_No"].duplicated().sum()))
print("Date range:", raw["Order_Date"].min().date(), "to", raw["Order_Date"].max().date())
print("\nMissing values:\n", raw.isnull().sum()[raw.isnull().sum() > 0])
print("Orders with zero revenue:", int((raw["Revenue"] == 0).sum()),
      "| of which also missing gross margin:", int(((raw["Revenue"] == 0) & raw["Gross margin"].isnull()).sum()))

# Zero-revenue orders are not real sales (and have no margin), so they are removed
df = raw[raw["Revenue"] > 0].copy().reset_index(drop=True)
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["LogQuantity"] = np.log(df["Quantity"])
print("Clean shape:", df.shape)

# Time-based split: train on 2012-2013, test on 2014 (the model must predict a FUTURE year)
train = df[df["Year"] <= 2013].copy()
test = df[df["Year"] == 2014].copy()
THRESH = train["Revenue"].quantile(0.75)               # 'high value' = top quarter of TRAINING orders
for d_ in (train, test, df):
    d_["HighValue"] = (d_["Revenue"] >= THRESH).astype(int)
print("\nHigh-value threshold (75th percentile of 2012-2013 revenue): %.2f" % THRESH)
print("High-value share: train %.1f%%, test %.1f%%" % (100 * train.HighValue.mean(), 100 * test.HighValue.mean()))
print("Train rows:", len(train), " Test rows:", len(test))
y_tr, y_te = train["HighValue"].to_numpy(), test["HighValue"].to_numpy()

# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =====================================================================
print("\nRevenue summary:\n", df["Revenue"].describe().round(0))
print("Revenue skewness: %.2f; skewness of log revenue: %.2f" % (df["Revenue"].skew(), np.log(df["Revenue"]).skew()))
print("\nOrders and revenue by year:\n", df.groupby("Year")["Revenue"].agg(["count", "sum", "mean"]).round(0))
print("\nOrder mode by year (share of orders, %):\n",
      (pd.crosstab(df["Year"], df["Order_Mode"], normalize="index") * 100).round(1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Revenue"], bins=50, ax=ax[0]); ax[0].set_title("Fig 1a. Order revenue")
sns.histplot(np.log(df["Revenue"]), bins=50, ax=ax[1]); ax[1].set_title("Fig 1b. Log order revenue")
plt.tight_layout(); plt.savefig("figures/fig1_revenue_distribution.png", dpi=150); plt.close()

monthly = df.groupby(["Year", "Month"]).agg(Orders=("Revenue", "size"), Revenue=("Revenue", "sum")).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(data=monthly, x="Month", y="Orders", hue="Year", palette="tab10", ax=ax[0]); ax[0].set_title("Fig 2a. Orders per month")
sns.lineplot(data=monthly, x="Month", y="Revenue", hue="Year", palette="tab10", ax=ax[1]); ax[1].set_title("Fig 2b. Revenue per month")
plt.tight_layout(); plt.savefig("figures/fig2_monthly.png", dpi=150); plt.close()

(pd.crosstab(df["Year"], df["Order_Mode"], normalize="index") * 100).plot(kind="bar", stacked=True, figsize=(8, 4), colormap="tab10")
plt.title("Fig 3. Order channel mix by year (%)"); plt.ylabel("%"); plt.legend(fontsize=7, bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.savefig("figures/fig3_channel_mix.png", dpi=150); plt.close()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for a, col in zip(ax.ravel(), ["Retailer_Type", "Order_Mode", "Product_Category", "Retailer_Country"]):
    train.groupby(col)["HighValue"].mean().sort_values().plot(kind="barh", ax=a)
    a.set_title("Fig 4. High-value rate by %s" % col); a.set_xlabel("Share of orders that are high value")
plt.tight_layout(); plt.savefig("figures/fig4_high_value_rates.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(df["Quantity"], df["Revenue"], s=4, alpha=.3); plt.xscale("log"); plt.yscale("log")
plt.xlabel("Quantity"); plt.ylabel("Revenue"); plt.title("Fig 5. Revenue vs quantity (log-log)")
plt.tight_layout(); plt.savefig("figures/fig5_quantity_revenue.png", dpi=150); plt.close()

# Statistical tests on the training years
print("\nSTATISTICAL TESTS (training years): association with HighValue")
rows = []
for col in ["Retailer_Country", "Retailer_Type", "Order_Mode", "Product_Category", "Product_Type"]:
    ct = pd.crosstab(train[col], train["HighValue"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    cramer = np.sqrt(chi2 / (len(train) * (min(ct.shape) - 1)))
    rows.append([col, ct.shape[0], chi2, p, cramer])
print(pd.DataFrame(rows, columns=["Variable", "Levels", "Chi2", "p", "Cramers_V"]).round(4).to_string(index=False))
u = stats.mannwhitneyu(train.loc[train.HighValue == 1, "Quantity"], train.loc[train.HighValue == 0, "Quantity"])
print("Quantity, high vs not high (Mann-Whitney): p=%.2e; medians %d vs %d" % (
    u.pvalue, train.loc[train.HighValue == 1, "Quantity"].median(), train.loc[train.HighValue == 0, "Quantity"].median()))
rho, p = stats.spearmanr(train["Quantity"], train["Revenue"])
print("Spearman correlation quantity vs revenue: rho=%.3f p=%.2e" % (rho, p))
H, p = stats.kruskal(*[g["Revenue"].values for _, g in train.groupby("Order_Mode")])
print("Kruskal-Wallis revenue by order mode: H=%.1f p=%.2e" % (H, p))
H, p = stats.kruskal(*[g["Revenue"].values for _, g in train.groupby("Retailer_Type")])
print("Kruskal-Wallis revenue by retailer type: H=%.1f p=%.2e" % (H, p))
unit_price = df["Revenue"] / df["Quantity"]
cv = (unit_price.groupby(df["Product"]).std() / unit_price.groupby(df["Product"]).mean())
print("Unit price (Revenue/Quantity) coefficient of variation within a product: median %.3f" % cv.median())

# =====================================================================
# 3. FEATURE SETS AND PREPROCESSING
# =====================================================================
# Gross margin is excluded: it is only known after the sale (and is derived from price and cost)
FEATURE_SETS = {
    "A: account and channel": (["Retailer_Country", "Retailer_Type", "Order_Mode"], ["Month"]),
    "B: A + product type + quantity": (["Retailer_Country", "Retailer_Type", "Order_Mode",
                                        "Product_Category", "Product_Type"], ["Month", "LogQuantity"]),
    "C: B + product SKU": (["Retailer_Country", "Retailer_Type", "Order_Mode", "Product_Category",
                            "Product_Type", "Product"], ["Month", "LogQuantity"]),
}
def preprocessor(cats, nums):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                              ("num", StandardScaler(), nums)])

MODELS = {
    "Majority baseline": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced",
                                                    random_state=SEED, n_jobs=-1),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}
def make_pipe(model_name, set_name):
    cats, nums = FEATURE_SETS[set_name]
    return Pipeline([("prep", preprocessor(cats, nums)), ("model", MODELS[model_name]())])

# =====================================================================
# 4. MODEL SELECTION (5-fold stratified CV on the training years)
# =====================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for sname, (cats, nums) in FEATURE_SETS.items():
    for mname in MODELS:
        if mname == "Majority baseline" and not sname.startswith("A"):
            continue
        res = cross_validate(make_pipe(mname, sname), train[cats + nums], y_tr, cv=cv, n_jobs=1,
                             scoring={"auc": "roc_auc", "ap": "average_precision", "f1": "f1"})
        rows.append([sname, mname, res["test_auc"].mean(), res["test_ap"].mean(), res["test_f1"].mean(), res["test_f1"].std()])
cvtab = pd.DataFrame(rows, columns=["Feature set", "Model", "ROC_AUC", "Avg_Precision", "F1", "F1_sd"])
print("\n" + "=" * 70 + "\nMODEL COMPARISON (5-fold CV, training years)\n" + "=" * 70)
print(cvtab.round(3).to_string(index=False))

setB = "B: A + product type + quantity"
candidates = cvtab[(cvtab["Feature set"] == setB) & (cvtab["Model"] != "Majority baseline")]
best_name = candidates.loc[candidates["Avg_Precision"].idxmax(), "Model"]
print("\nBest model on feature set B by average precision:", best_name)

# =====================================================================
# 5. HYPERPARAMETER TUNING
# =====================================================================
GRIDS = {
    "Random forest": {"model__max_depth": [8, 16, None], "model__min_samples_leaf": [1, 3, 8],
                      "model__max_features": ["sqrt", 0.5]},
    "Gradient boosting": {"model__learning_rate": [0.03, 0.1], "model__max_depth": [3, None],
                          "model__max_leaf_nodes": [15, 31], "model__l2_regularization": [0.0, 1.0]},
    "Logistic regression": {"model__C": [0.01, 0.1, 1, 10]},
}
cats, nums = FEATURE_SETS[setB]
gs = GridSearchCV(make_pipe(best_name, setB), GRIDS[best_name], cv=cv, scoring="average_precision", n_jobs=1)
gs.fit(train[cats + nums], y_tr)
print("\nBest hyperparameters:", gs.best_params_)
print("CV average precision: default %.4f -> tuned %.4f" % (
    candidates.loc[candidates.Model == best_name, "Avg_Precision"].iloc[0], gs.best_score_))
final = gs.best_estimator_

# =====================================================================
# 6. THRESHOLD CHOICE (out-of-fold on the training years) AND HOLD-OUT EVALUATION (2014)
# =====================================================================
oof = cross_val_predict(gs.best_estimator_, train[cats + nums], y_tr, cv=cv, method="predict_proba")[:, 1]
ths = np.arange(0.10, 0.91, 0.05)
f1s = [f1_score(y_tr, (oof >= t).astype(int)) for t in ths]
best_t = float(ths[int(np.argmax(f1s))])
print("Threshold maximising out-of-fold F1: %.2f (F1=%.3f)" % (best_t, max(f1s)))

p_te = final.predict_proba(test[cats + nums])[:, 1]
print("\n" + "=" * 70 + "\nHOLD-OUT RESULTS: trained on 2012-2013, tested on 2014\n" + "=" * 70)
rows = []
base = make_pipe("Majority baseline", "A: account and channel").fit(train, y_tr)
rows.append(["Majority baseline", 0.5, roc_auc_score(y_te, base.predict_proba(test)[:, 1]), average_precision_score(y_te, base.predict_proba(test)[:, 1]),
             0, 0, 0, (base.predict(test) == y_te).mean()])
qty_nums = ["LogQuantity"]
qmodel = Pipeline([("prep", StandardScaler()),
                   ("model", LogisticRegression(class_weight="balanced"))]).fit(train[qty_nums], y_tr)
pq = qmodel.predict_proba(test[qty_nums])[:, 1]
rows.append(["Quantity-only logistic regression", 0.5, roc_auc_score(y_te, pq), average_precision_score(y_te, pq),
             precision_score(y_te, pq >= .5), recall_score(y_te, pq >= .5), f1_score(y_te, pq >= .5), ((pq >= .5) == y_te).mean()])
for t in [0.5, best_t]:
    pred = (p_te >= t).astype(int)
    rows.append(["Tuned %s (threshold %.2f)" % (best_name, t), t, roc_auc_score(y_te, p_te), average_precision_score(y_te, p_te),
                 precision_score(y_te, pred), recall_score(y_te, pred), f1_score(y_te, pred), (pred == y_te).mean()])
hold = pd.DataFrame(rows, columns=["Model", "Threshold", "ROC_AUC", "Avg_Precision", "Precision", "Recall", "F1", "Accuracy"])
print(hold.round(3).to_string(index=False))

pred = (p_te >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print("\nConfusion matrix at threshold %.2f: TN=%d FP=%d FN=%d TP=%d" % (best_t, tn, fp, fn, tp))
print("Of 100 orders flagged as high value, about %d really are (precision)" % round(100 * tp / (tp + fp)))
print("The model finds %d%% of all high-value orders (recall)" % round(100 * tp / (tp + fn)))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for name, p_, in [("Tuned " + best_name, p_te), ("Quantity only", pq)]:
    fpr, tpr, _ = roc_curve(y_te, p_); ax[0].plot(fpr, tpr, label="%s (AUC %.3f)" % (name, roc_auc_score(y_te, p_)))
    pr, rc, _ = precision_recall_curve(y_te, p_); ax[1].plot(rc, pr, label="%s (AP %.3f)" % (name, average_precision_score(y_te, p_)))
ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Fig 6a. ROC curve (2014)"); ax[0].legend(fontsize=8)
ax[1].set(xlabel="Recall", ylabel="Precision", title="Fig 6b. Precision-recall curve (2014)"); ax[1].legend(fontsize=8)
ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["Not high", "High"]).plot(ax=ax[2], colorbar=False)
ax[2].set_title("Fig 6c. Confusion matrix (threshold %.2f)" % best_t)
plt.tight_layout(); plt.savefig("figures/fig6_evaluation.png", dpi=150); plt.close()

# Does performance hold across groups?
test_eval = test.assign(pred=pred)
print("\nRecall of high-value orders by order mode (2014):")
print(test_eval[test_eval.HighValue == 1].groupby("Order_Mode")["pred"].agg(["mean", "count"]).round(3))

# =====================================================================
# 7. FEATURE IMPORTANCE (permutation importance on the 2014 hold-out)
# =====================================================================
pi = permutation_importance(final, test[cats + nums], y_te, n_repeats=10, random_state=SEED,
                            scoring="average_precision")
imp = pd.Series(pi.importances_mean, index=cats + nums).sort_values(ascending=False)
print("\nPermutation importance (drop in average precision):\n", imp.round(4))
imp.iloc[::-1].plot(kind="barh", figsize=(7, 4), title="Fig 7. Permutation importance (2014 hold-out)")
plt.xlabel("Drop in average precision"); plt.tight_layout(); plt.savefig("figures/fig7_importance.png", dpi=150); plt.close()