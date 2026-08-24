"""
Agricultural Variety Classification & Comparative Visualization Analysis
Models: RBF-SVM vs. Random Forest
Datasets: UCI Rice (ID: 545) & UCI Dry Bean (ID: 602)
Course: CA6002 Data Visualisation

Outputs:
    High-resolution publication-quality PNG charts and CSV result summaries in 'agricultural_model_outputs/'
"""

import os
import time
import warnings
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import seaborn as sns
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================== Global Configurations ==============================
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
FULL_GRID_SEARCH = False
OUTPUT_DIR = "agricultural_model_outputs"
DATA_CACHE_DIR = "data_cache"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Aesthetic Styling and English Palettes
PALETTE_MODELS = {
    "RBF-SVM": "#1E40AF",       # Deep Royal Blue
    "Random Forest": "#D97706"  # Amber Orange
}

COLORS = {
    "Primary": "#2563EB",
    "Secondary": "#059669",
    "Accent": "#7C3AED",
    "Highlight": "#DC2626",     # Crimson Red
    "NeutralDark": "#1F2937",
    "NeutralLight": "#F3F4F6",
}

# Set professional scientific publication styling
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#D1D5DB",
    "axes.linewidth": 1.0,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "axes.labelpad": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def save_figure(file_name: str):
    """Save current figure in high DPI and close plot."""
    path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Saved Figure] -> {path}")


# ============================== Data Ingestion & Preprocessing ==============================
def load_uci_dataset(dataset_id: int, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from local cache or fetch from UCI Repository."""
    cache_path_x = os.path.join(DATA_CACHE_DIR, f"{dataset_name.lower()}_x.csv")
    cache_path_y = os.path.join(DATA_CACHE_DIR, f"{dataset_name.lower()}_y.csv")

    if os.path.exists(cache_path_x) and os.path.exists(cache_path_y):
        print(f"Loading {dataset_name} from local cache...")
        X = pd.read_csv(cache_path_x)
        y = pd.read_csv(cache_path_y).iloc[:, 0]
    else:
        print(f"Fetching {dataset_name} (ID: {dataset_id}) from UCI Repository...")
        try:
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features.copy()
            y = dataset.data.targets.copy()
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            X.to_csv(cache_path_x, index=False)
            y.to_csv(cache_path_y, index=False)
        except Exception as err:
            print(f"Error fetching dataset {dataset_name}: {err}")
            raise

    X.columns = [str(c).strip().replace(" ", "_") for c in X.columns]
    y = y.astype(str).str.strip()
    y.name = "Class"

    print(f"  -> {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, {y.nunique()} classes.")
    return X, y


def clean_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Coerce numerical values, impute missing values if any, and deduplicate."""
    X_clean = X.apply(pd.to_numeric, errors="coerce")
    if X_clean.isna().sum().sum() > 0:
        X_clean = X_clean.fillna(X_clean.median())

    full = X_clean.copy()
    full["Class"] = y.values
    dup_count = int(full.duplicated().sum())
    if dup_count > 0:
        print(f"  -> {dataset_name}: Removed {dup_count} duplicate records.")
        full = full.drop_duplicates().reset_index(drop=True)

    y_clean = full.pop("Class")
    return full, y_clean


def inspect_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str):
    """Print statistical quality summary."""
    print(f"\n--- [{dataset_name} Data Quality Summary] ---")
    print(f"Sample Count: {len(X)} | Feature Count: {X.shape[1]}")
    dist = (y.value_counts(normalize=True) * 100).round(2)
    print("Class Proportions (%):")
    for cls, pct in dist.items():
        print(f"  - {cls}: {pct}% (N={sum(y==cls)})")


# ============================== Exploratory Data Visualizations ==============================
def draw_confidence_ellipse(ax, data: np.ndarray, color: str, n_std: float = 2.0):
    """Draw a 2D confidence ellipse representing the covariance of the class cluster."""
    if len(data) < 3:
        return
    cov = np.cov(data, rowvar=False)
    if np.isnan(cov).any() or np.isinf(cov).any():
        return
    
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    pearson = np.clip(pearson, -1.0, 1.0)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(max(1e-12, 1 - pearson))
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=color, alpha=0.12, edgecolor=color, linestyle="--", linewidth=1.5)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(data[:, 0])
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(data[:, 1])

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_class_distribution(y: pd.Series, dataset_name: str):
    """Plot sample count and percentage breakdown for each agricultural variety."""
    plt.figure(figsize=(9, 5.5))
    counts = y.value_counts()
    total = len(y)
    palette = sns.color_palette("Blues_r", n_colors=len(counts))
    
    ax = sns.barplot(x=counts.index, y=counts.values, palette=palette, edgecolor="black", linewidth=0.6)
    ax.set_title(f"{dataset_name} Dataset: Variety Distribution", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Agricultural Variety", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Sample Count (N)", fontsize=12, fontweight="semibold")
    
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{int(height):,}\n({pct:.1f}%)", 
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points",
                    fontsize=10, fontweight="medium")
    
    ax.set_ylim(0, max(counts.values) * 1.18)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_class_distribution.png")


def plot_feature_histograms(X: pd.DataFrame, dataset_name: str):
    """Plot distribution histograms with KDE curves for all morphological features."""
    features = list(X.columns)
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    axes = np.asarray(axes).flatten()

    for i, col in enumerate(features):
        sns.histplot(X[col], kde=True, ax=axes[i], color=COLORS["Primary"], 
                     alpha=0.45, edgecolor="none", line_kws={"linewidth": 1.5})
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Feature Value", fontsize=9)
        axes[i].set_ylabel("Frequency", fontsize=9)
        axes[i].tick_params(axis='both', which='major', labelsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{dataset_name} Dataset: Morphological Feature Distributions", 
                 fontsize=16, fontweight="bold", y=1.01)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_feature_histograms.png")


def plot_correlation_heatmap(X: pd.DataFrame, dataset_name: str):
    """Plot masked correlation matrix for morphological attributes."""
    plt.figure(figsize=(11, 9))
    corr = X.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                     vmin=-1, vmax=1, square=True, linewidths=0.7, 
                     cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
                     annot_kws={"size": 8 if len(X.columns) > 10 else 10})
    ax.set_title(f"{dataset_name} Dataset: Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_correlation_heatmap.png")


def plot_pca_projection(X: pd.DataFrame, y: pd.Series, dataset_name: str):
    """Plot 2D PCA projection overlaying 95% confidence ellipses for class clustering."""
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projected = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(projected, columns=["PC1", "PC2"])
    df_pca["Class"] = y.values
    var = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(10, 7.5))
    ax = plt.gca()
    unique_classes = sorted(y.unique())
    palette = sns.color_palette("tab10" if len(unique_classes) > 2 else ["#2563EB", "#D97706"], n_colors=len(unique_classes))

    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Class", hue_order=unique_classes,
                    palette=palette, alpha=0.55, s=40, edgecolor="w", linewidth=0.3)

    for i, cls in enumerate(unique_classes):
        cls_data = df_pca[df_pca["Class"] == cls][["PC1", "PC2"]].values
        if len(cls_data) > 2:
            draw_confidence_ellipse(ax, cls_data, palette[i], n_std=2.0)
            # Add centroid text badge
            cx, cy = np.mean(cls_data[:, 0]), np.mean(cls_data[:, 1])
            ax.text(cx, cy, cls, fontsize=9, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=palette[i], alpha=0.9, lw=1.2))

    ax.set_xlabel(f"Principal Component 1 (PC1: {var[0]:.1f}% Variance)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(f"Principal Component 2 (PC2: {var[1]:.1f}% Variance)", fontsize=12, fontweight="semibold")
    ax.set_title(f"{dataset_name} Dataset: 2D PCA Projection with 95% Confidence Ellipses", 
                 fontsize=14, fontweight="bold", pad=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Variety", frameon=True)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_pca_projection.png")


# ============================== Model Pipeline & Grid Search ==============================
def get_model_configurations(full_grid: bool = False) -> Dict[str, Dict[str, Any]]:
    """Define model pipelines and hyperparameter search grids."""
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
    ])

    if full_grid:
        svm_grid = {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.001, 0.01, 0.1, 1]}
        rf_grid = {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 10, 15, 25],
            "model__min_samples_leaf": [1, 2, 4],
        }
    else:
        svm_grid = {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]}
        rf_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 15],
            "model__min_samples_leaf": [1, 2],
        }

    return {
        "RBF-SVM": {"pipeline": svm, "parameter_grid": svm_grid},
        "Random Forest": {"pipeline": rf, "parameter_grid": rf_grid},
    }


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, dataset_name: str,
                              configurations: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Train models via GridSearchCV, evaluate on test set, and compute fold metrics."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    rows, fitted = [], {}
    per_class_records = []

    print(f"\n[Training Models on {dataset_name}]")
    for name, config in configurations.items():
        print(f"  -> Training {name}...")
        search = GridSearchCV(config["pipeline"], config["parameter_grid"], scoring="accuracy",
                              cv=cv, n_jobs=-1, refit=True)
        start = time.perf_counter()
        search.fit(X_train, y_train)
        duration = time.perf_counter() - start

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        # 5-Fold cross-validation scores of the best estimator on training set
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

        test_acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)

        metrics = {
            "CV Accuracy": search.best_score_,
            "CV Std": float(np.std(cv_scores)),
            "Test Accuracy": test_acc,
            "Balanced Accuracy": bal_acc,
            "Macro Precision": macro_p,
            "Macro Recall": macro_r,
            "Macro F1": macro_f1,
            "Time": duration,
        }

        # Per-class metrics
        class_precisions = precision_score(y_test, y_pred, average=None, zero_division=0)
        class_recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
        class_f1s = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        for idx, cls_label in enumerate(le.classes_):
            per_class_records.append({
                "Dataset": dataset_name,
                "Model": name,
                "Variety": cls_label,
                "Precision": class_precisions[idx],
                "Recall": class_recalls[idx],
                "F1-Score": class_f1s[idx],
                "Support": int(sum(y_test == idx))
            })

        rows.append({"Dataset": dataset_name, "Model": name, **metrics})
        fitted[name] = {
            "model": best_model,
            "grid_search": search,
            "cv_scores": cv_scores,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "label_encoder": le,
        }
        print(f"     Best Params: {search.best_params_}")
        print(f"     Test Accuracy: {test_acc:.4f} | Macro F1: {macro_f1:.4f} | CV Std: {np.std(cv_scores):.4f}")

    return pd.DataFrame(rows), fitted, pd.DataFrame(per_class_records)


# ============================== AI Algorithm Design & Tuning Visualizations ==============================
def plot_svm_parameter_heatmap(fitted_models: Dict[str, Any], dataset_name: str):
    """Plot RBF-SVM hyperparameter validation surface (C vs Gamma)."""
    res = fitted_models["RBF-SVM"]["grid_search"]
    results = pd.DataFrame(res.cv_results_)

    results["param_model__C"] = results["param_model__C"].astype(float)
    results["param_model__gamma"] = results["param_model__gamma"].astype(str)

    pivot = results.pivot(index="param_model__gamma", columns="param_model__C", values="mean_test_score")

    plt.figure(figsize=(9, 6.5))
    ax = sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", 
                     cbar_kws={"label": "Mean 5-Fold CV Accuracy"}, linewidths=1.0)
    
    # Highlight the best parameter combination
    best_c = float(res.best_params_["model__C"])
    best_gamma = str(res.best_params_["model__gamma"])
    
    ax.set_title(f"{dataset_name} Dataset: RBF-SVM Hyperparameter Grid Search", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Regularization Parameter (C)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Kernel Coefficient (Gamma)", fontsize=12, fontweight="semibold")
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_svm_parameter_heatmap.png")


def plot_rf_validation_curve(X: pd.DataFrame, y: pd.Series, dataset_name: str):
    """Plot Random Forest validation curve over tree depth (Bias-Variance tradeoff)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    param_range = [2, 4, 8, 12, 16, 20, 30]
    param_labels = [str(p) for p in param_range]

    train_scores, test_scores = validation_curve(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"),
        X, y_enc, param_name="max_depth", param_range=param_range,
        cv=5, scoring="accuracy", n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(9, 5.5))
    plt.plot(param_labels, train_mean, label="Training Accuracy", marker="o", color=PALETTE_MODELS["RBF-SVM"], linewidth=2)
    plt.fill_between(param_labels, train_mean - train_std, train_mean + train_std, alpha=0.15, color=PALETTE_MODELS["RBF-SVM"])

    plt.plot(param_labels, test_mean, label="5-Fold CV Accuracy", marker="s", color=PALETTE_MODELS["Random Forest"], linewidth=2)
    plt.fill_between(param_labels, test_mean - test_std, test_mean + test_std, alpha=0.15, color=PALETTE_MODELS["Random Forest"])

    plt.title(f"{dataset_name} Dataset: Random Forest Max Depth Validation Curve", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Maximum Tree Depth (max_depth)", fontsize=12, fontweight="semibold")
    plt.ylabel("Classification Accuracy", fontsize=12, fontweight="semibold")
    plt.legend(loc="lower right", frameon=True)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_rf_validation_curve.png")


def plot_learning_curves(fitted_models: Dict[str, Any], dataset_name: str):
    """Plot sample-efficiency learning curves for both RBF-SVM and Random Forest."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    train_sizes = np.linspace(0.1, 1.0, 6)

    for ax, (model_name, res) in zip(axes, fitted_models.items()):
        estimator = res["model"]
        X_train, y_train = res["X_train"], res["y_train"]

        t_sizes, train_scores, val_scores = learning_curve(
            estimator, X_train, y_train, train_sizes=train_sizes,
            cv=5, scoring="accuracy", n_jobs=-1, random_state=RANDOM_STATE
        )

        t_mean = np.mean(train_scores, axis=1)
        t_std = np.std(train_scores, axis=1)
        v_mean = np.mean(val_scores, axis=1)
        v_std = np.std(val_scores, axis=1)

        ax.plot(t_sizes, t_mean, "o-", color=COLORS["Primary"], label="Training Score", linewidth=2)
        ax.fill_between(t_sizes, t_mean - t_std, t_mean + t_std, alpha=0.15, color=COLORS["Primary"])

        ax.plot(t_sizes, v_mean, "s-", color=COLORS["Secondary"], label="Cross-Validation Score", linewidth=2)
        ax.fill_between(t_sizes, v_mean - v_std, v_mean + v_std, alpha=0.15, color=COLORS["Secondary"])

        ax.set_title(f"{model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Sample Size (N)", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Accuracy Score", fontsize=11, fontweight="semibold")
        ax.legend(loc="lower right", frameon=True)

    fig.suptitle(f"{dataset_name} Dataset: Learning Curves (Sample Efficiency & Convergence)", 
                 fontsize=15, fontweight="bold", y=1.02)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_learning_curves.png")


def plot_rf_feature_importance(fitted_models: Dict[str, Any], X: pd.DataFrame, dataset_name: str, top_n: int = 12):
    """Plot Random Forest feature importance ranked by Gini impurity reduction."""
    rf = fitted_models["Random Forest"]["model"].named_steps["model"]
    importances = rf.feature_importances_
    
    data = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    data = data.nlargest(top_n, "Importance").sort_values("Importance", ascending=True)

    plt.figure(figsize=(9, 6.5))
    bars = plt.barh(data["Feature"], data["Importance"], color=PALETTE_MODELS["Random Forest"], 
                    alpha=0.85, edgecolor="black", linewidth=0.5)
    
    # Add numerical labels
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", 
                 va="center", ha="left", fontsize=9, fontweight="medium")

    plt.xlim(0, max(data["Importance"]) * 1.18)
    plt.title(f"{dataset_name} Dataset: Random Forest Feature Importance", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight="semibold")
    plt.ylabel("Morphological Feature", fontsize=11, fontweight="semibold")
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_rf_feature_importance.png")


# ============================== Evaluation & Diagnostics Visualizations ==============================
def plot_confusion_matrices(model_results: Dict[str, Any], dataset_name: str):
    """Plot normalized test set confusion matrices side-by-side with counts and percentages."""
    fig, axes = plt.subplots(1, len(model_results), figsize=(15, 6.5))
    axes = np.atleast_1d(axes)

    for ax, (name, res) in zip(axes, model_results.items()):
        cm = confusion_matrix(res["y_test"], res["y_pred"])
        cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        labels = [f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_pct.flatten())]
        labels = np.array(labels).reshape(cm.shape)

        sns.heatmap(cm_pct, annot=labels, fmt="", cmap="Blues", ax=ax, cbar=False,
                    vmin=0, vmax=1, linewidths=0.8, linecolor="white",
                    xticklabels=res["label_encoder"].classes_,
                    yticklabels=res["label_encoder"].classes_,
                    annot_kws={"size": 8 if len(cm) > 4 else 11, "fontweight": "medium"})
        
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold")
        ax.set_ylabel("True Class", fontsize=11, fontweight="semibold")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle(f"{dataset_name} Dataset: Test Confusion Matrices (Count & Recall %)", 
                 fontsize=15, fontweight="bold", y=1.02)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_confusion_matrices.png")


def plot_per_class_metrics(per_class_df: pd.DataFrame, dataset_name: str):
    """Plot Precision, Recall, and F1-Score breakdown for each agricultural variety."""
    df_sub = per_class_df[per_class_df["Dataset"] == dataset_name]
    
    # Melt for multi-metric grouped bar plotting
    melted = df_sub.melt(id_vars=["Model", "Variety"], value_vars=["Precision", "Recall", "F1-Score"], 
                         var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=melted, x="Variety", y="Score", hue="Metric",
                     palette=["#2563EB", "#059669", "#D97706"], edgecolor="black", linewidth=0.5)
    
    ax.set_ylim(0.60, 1.03)
    ax.set_title(f"{dataset_name} Dataset: Per-Variety Classification Performance", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Agricultural Variety", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Metric Score", fontsize=12, fontweight="semibold")
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower right", frameon=True)
    save_figure(f"{dataset_name.lower().replace(' ', '_')}_per_class_metrics.png")


def plot_error_space_pca(fitted_models: Dict[str, Any], dataset_name: str):
    """Diagnose misclassified test samples projected onto PCA feature space."""
    for model_name, res in fitted_models.items():
        X_test_scaled = StandardScaler().fit_transform(res["X_test"])
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        projected = pca.fit_transform(X_test_scaled)
        var = pca.explained_variance_ratio_ * 100

        df = pd.DataFrame(projected, columns=["PC1", "PC2"])
        df["True"] = res["label_encoder"].inverse_transform(res["y_test"])
        df["Pred"] = res["label_encoder"].inverse_transform(res["y_pred"])
        df["Correct"] = df["True"] == df["Pred"]

        err_count = sum(~df["Correct"])
        err_rate = (err_count / len(df)) * 100

        plt.figure(figsize=(10, 7.5))
        
        # Plot correctly classified samples in subtle transparent tones
        sns.scatterplot(data=df[df["Correct"]], x="PC1", y="PC2", hue="True",
                        alpha=0.35, s=35, palette="tab10" if df["True"].nunique() > 2 else "Blues",
                        legend=True, edgecolor="none")

        # Highlight misclassified samples with prominent markers
        errors = df[~df["Correct"]]
        if not errors.empty:
            plt.scatter(errors["PC1"], errors["PC2"], color=COLORS["Highlight"],
                        marker="X", s=85, label=f"Misclassified (N={err_count})",
                        edgecolor="black", linewidth=0.6, zorder=5)

        plt.title(f"{dataset_name} Dataset: {model_name} Error Space Analysis\n(Error Rate: {err_rate:.2f}%)", 
                  fontsize=14, fontweight="bold", pad=12)
        plt.xlabel(f"Principal Component 1 (PC1: {var[0]:.1f}%)", fontsize=11, fontweight="semibold")
        plt.ylabel(f"Principal Component 2 (PC2: {var[1]:.1f}%)", fontsize=11, fontweight="semibold")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Legend", frameon=True)

        safe_name = f"{dataset_name}_{model_name}_error_pca".lower().replace(" ", "_").replace("-", "_")
        save_figure(f"{safe_name}.png")


def plot_cv_stability_comparison(fitted_rice: Dict[str, Any], fitted_bean: Dict[str, Any]):
    """Plot cross-validation score distributions across folds to compare model stability."""
    records = []
    for dname, models in [("Rice (2-Class)", fitted_rice), ("Dry Bean (7-Class)", fitted_bean)]:
        for mname, res in models.items():
            for score in res["cv_scores"]:
                records.append({"Dataset": dname, "Model": mname, "CV Accuracy": score})

    df_cv = pd.DataFrame(records)
    plt.figure(figsize=(9, 5.5))
    ax = sns.boxplot(data=df_cv, x="Dataset", y="CV Accuracy", hue="Model",
                     palette=PALETTE_MODELS, width=0.5, boxprops=dict(alpha=0.85))
    sns.stripplot(data=df_cv, x="Dataset", y="CV Accuracy", hue="Model",
                  palette=PALETTE_MODELS, dodge=True, color="black", alpha=0.6, size=6, jitter=0.1)

    # Clean legend duplicate handles
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="lower left", title="Model", frameon=True)

    ax.set_title("Cross-Validation Score Stability across 5 Folds", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset & Complexity", fontsize=12, fontweight="semibold")
    ax.set_ylabel("5-Fold CV Accuracy", fontsize=12, fontweight="semibold")
    save_figure("model_cv_stability_comparison.png")


def plot_model_comparison(results: pd.DataFrame, metric: str):
    """Plot benchmark comparison across datasets for a specific performance metric."""
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(data=results, x="Dataset", y=metric, hue="Model", 
                     palette=PALETTE_MODELS, edgecolor="black", linewidth=0.6)
    
    ax.set_title(f"Model Benchmark Comparison: {metric}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dataset", fontsize=12, fontweight="semibold")
    ax.set_ylabel(metric, fontsize=12, fontweight="semibold")

    ymin = results[metric].min() * 0.94
    ax.set_ylim(ymin, 1.02 if metric != "Time" else results[metric].max() * 1.15)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f" if metric != "Time" else "%.1fs", padding=3, fontsize=10, fontweight="medium")

    plt.legend(loc="lower right" if metric != "Time" else "upper left", frameon=True)
    save_figure(f"model_comparison_{metric.lower().replace(' ', '_')}.png")


# ============================== Main Pipeline Execution ==============================
def main():
    print("=========================================================================")
    print("  Agricultural Variety Classification & Visual Analytics Pipeline")
    print("=========================================================================\n")

    # 1. Load Data
    X_rice, y_rice = load_uci_dataset(545, "Rice")
    X_bean, y_bean = load_uci_dataset(602, "Dry Bean")

    # Clean Data
    X_rice, y_rice = clean_dataset(X_rice, y_rice, "Rice")
    X_bean, y_bean = clean_dataset(X_bean, y_bean, "Dry Bean")

    # 2. Exploratory Data Visualizations
    print("\n--- Generating Exploratory Visualizations ---")
    for X, y, name in [(X_rice, y_rice, "Rice"), (X_bean, y_bean, "Dry Bean")]:
        inspect_dataset(X, y, name)
        plot_class_distribution(y, name)
        plot_feature_histograms(X, name)
        plot_correlation_heatmap(X, name)
        plot_pca_projection(X, y, name)

    # 3. Model Training & Parameter Optimization
    print("\n--- Training & Optimizing Models ---")
    configs = get_model_configurations(FULL_GRID_SEARCH)
    rice_res, rice_models, rice_per_class = train_and_evaluate_models(X_rice, y_rice, "Rice", configs)
    bean_res, bean_models, bean_per_class = train_and_evaluate_models(X_bean, y_bean, "Dry Bean", configs)

    all_results = pd.concat([rice_res, bean_res], ignore_index=True)
    all_per_class = pd.concat([rice_per_class, bean_per_class], ignore_index=True)

    all_results.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)
    all_per_class.to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)
    print(f"\nSaved metric tables to '{OUTPUT_DIR}/'.")

    # 4. Deep Visual Analytics & Explainability
    print("\n--- Generating Model Design & Diagnostic Visualizations ---")
    for models_dict, X, y, name, per_class_df in [
        (rice_models, X_rice, y_rice, "Rice", rice_per_class),
        (bean_models, X_bean, y_bean, "Dry Bean", bean_per_class)
    ]:
        plot_confusion_matrices(models_dict, name)
        plot_svm_parameter_heatmap(models_dict, name)
        plot_rf_validation_curve(X, y, name)
        plot_learning_curves(models_dict, name)
        plot_rf_feature_importance(models_dict, X, name)
        plot_per_class_metrics(per_class_df, name)
        plot_error_space_pca(models_dict, name)

    # 5. Overall Comparative Synthesis
    print("\n--- Generating Overall Comparative Visualizations ---")
    for metric in ["Test Accuracy", "Macro F1", "Time"]:
        plot_model_comparison(all_results, metric)

    plot_cv_stability_comparison(rice_models, bean_models)

    print("\n=========================================================================")
    print(f"  Pipeline Finished Successfully! All English Visualizations in: {OUTPUT_DIR}")
    print("=========================================================================")


if __name__ == "__main__":
    main()
