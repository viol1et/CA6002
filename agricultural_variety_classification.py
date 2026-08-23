"""
农产品品种识别：RBF-SVM 与 Random Forest 对比
数据集：UCI Rice (ID 545) 与 Dry Bean (ID 602)

安装依赖：
    pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo

运行：
    python agricultural_variety_classification.py

输出：
    agricultural_model_outputs/ 目录下的 CSV 结果与 PNG 图表
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
CUDA_VISIBLE_DEVICE = 0

# ============================== 全局设置 ==============================
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
FULL_GRID_SEARCH = False  # True 会扩大参数搜索范围，但耗时显著增加
OUTPUT_DIR = "agricultural_model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def save_and_show_figure(file_name):
    """保存当前图表到输出目录并关闭图形。"""
    path = os.path.join(OUTPUT_DIR, file_name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"图表已保存：{path}")


# ============================== 数据加载与清理 ==============================
def load_uci_dataset(dataset_id, dataset_name):
    """根据 UCI ID 下载数据，并返回特征 X 与标签 y。"""
    print("\n" + "=" * 70)
    print(f"正在加载数据集：{dataset_name}")
    print("=" * 70)

    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X.columns = [str(c).strip().replace(" ", "_") for c in X.columns]
    y = y.astype(str).str.strip()
    y.name = "Class"

    print(f"样本数量：{X.shape[0]}")
    print(f"特征数量：{X.shape[1]}")
    print(f"类别数量：{y.nunique()}")
    print(f"缺失值总数：{X.isna().sum().sum()}")
    print("类别分布：")
    print(y.value_counts())
    return X, y


def clean_dataset(X, y, dataset_name):
    """数值化特征、中位数填补缺失值，并删除完全重复记录。"""
    X_clean = X.apply(pd.to_numeric, errors="coerce")
    if X_clean.isna().sum().sum() > 0:
        print(f"{dataset_name}：使用中位数填补缺失值。")
        X_clean = X_clean.fillna(X_clean.median())

    full = X_clean.copy()
    full["Class"] = y.values
    duplicate_count = int(full.duplicated().sum())
    if duplicate_count:
        print(f"{dataset_name}：删除 {duplicate_count} 条完全重复记录。")
        full = full.drop_duplicates().reset_index(drop=True)

    y_clean = full.pop("Class")
    print(f"{dataset_name} 清理后：{full.shape[0]} 个样本，{full.shape[1]} 个特征。")
    return full, y_clean


def inspect_dataset(X, y, dataset_name):
    """输出数据类型、描述性统计和类别比例。"""
    print("\n" + "=" * 70)
    print(f"{dataset_name} 数据质量检查")
    print("=" * 70)
    print("\n特征类型：")
    print(X.dtypes)
    print("\n描述性统计：")
    print(X.describe().T.round(3))
    print("\n类别比例：")
    print((y.value_counts(normalize=True).sort_index() * 100).round(2).astype(str) + "%")


# ============================== 探索性可视化 ==============================
def plot_class_distribution(y, dataset_name):
    counts = y.value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4C78A8")
    ax.set(title=f"{dataset_name}: Class Distribution", xlabel="Variety", ylabel="Number of Samples")
    ax.tick_params(axis="x", rotation=35)
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_class_distribution.png")


def plot_feature_histograms(X, dataset_name):
    features = list(X.columns)
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for i, feature in enumerate(features):
        sns.histplot(X[feature], bins=30, kde=True, ax=axes[i], color="#4C78A8")
        axes[i].set_title(feature)
    for i in range(len(features), len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle(f"{dataset_name}: Feature Distributions", fontsize=17, y=1.01)
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_feature_histograms.png")


def plot_correlation_heatmap(X, dataset_name):
    size = max(9, X.shape[1] * 0.75)
    plt.figure(figsize=(size, size * 0.75))
    sns.heatmap(X.corr(numeric_only=True), cmap="coolwarm", center=0, square=True, linewidths=0.4)
    plt.title(f"{dataset_name}: Feature Correlation Matrix")
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_correlation_heatmap.png")


def plot_pca_projection(X, y, dataset_name):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projected = pca.fit_transform(X_scaled)
    data = pd.DataFrame({"PC1": projected[:, 0], "PC2": projected[:, 1], "Class": y.values})
    variance = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=data, x="PC1", y="PC2", hue="Class", palette="tab10", alpha=0.65, s=35)
    plt.xlabel(f"PC1 ({variance[0]:.1f}% explained variance)")
    plt.ylabel(f"PC2 ({variance[1]:.1f}% explained variance)")
    plt.title(f"{dataset_name}: PCA Projection")
    plt.legend(title="Variety", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_pca_projection.png")


# ============================== 模型配置 ==============================
def get_model_configurations(full_grid=False):
    """创建模型 Pipeline 及其参数网格。"""
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        )),
    ])

    if full_grid:
        svm_grid = {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.001, 0.01, 0.1, 1],
        }
        rf_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 10, 20, 30],
            "model__max_features": ["sqrt", "log2"],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    else:
        svm_grid = {
            "model__C": [1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1],
        }
        rf_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 15, 25],
            "model__max_features": ["sqrt"],
            "model__min_samples_leaf": [1, 2, 4],
        }

    return {
        "RBF-SVM": {"pipeline": svm, "parameter_grid": svm_grid},
        "Random Forest": {"pipeline": rf, "parameter_grid": rf_grid},
    }


# ============================== 训练与评价 ==============================
def train_and_evaluate_models(X, y, dataset_name, configurations):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}

    for model_name, config in configurations.items():
        print("\n" + "=" * 65)
        print(f"数据集：{dataset_name}；模型：{model_name}")
        print("=" * 65)

        search = GridSearchCV(
            config["pipeline"], config["parameter_grid"], scoring="accuracy",
            cv=cv, n_jobs=-1, return_train_score=True, refit=True, verbose=1
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        best_model = search.best_estimator_
        start = time.perf_counter()
        y_pred = best_model.predict(X_test)
        prediction_time = time.perf_counter() - start

        metrics = {
            "CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        print("最佳参数：", search.best_params_)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print(f"训练及调参时间：{training_time:.2f} 秒")
        print(classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
        ))

        rows.append({
            "Dataset": dataset_name,
            "Model": model_name,
            **metrics,
            "Training Time": training_time,
            "Prediction Time": prediction_time,
            "Best Parameters": str(search.best_params_),
        })
        fitted[model_name] = {
            "model": best_model,
            "grid_search": search,
            "X_train": X_train.copy(),
            "X_test": X_test.copy(),
            "y_train": y_train.copy(),
            "y_test": y_test.copy(),
            "y_pred": y_pred.copy(),
            "label_encoder": label_encoder,
        }

    return pd.DataFrame(rows), fitted


# ============================== 模型结果可视化 ==============================
def plot_confusion_matrices(model_results, dataset_name):
    fig, axes = plt.subplots(1, len(model_results), figsize=(8 * len(model_results), 7))
    axes = np.atleast_1d(axes)
    for ax, (model_name, result) in zip(axes, model_results.items()):
        matrix = confusion_matrix(result["y_test"], result["y_pred"])
        display = ConfusionMatrixDisplay(matrix, display_labels=result["label_encoder"].classes_)
        display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d")
        ax.set_title(f"{dataset_name}: {model_name}")
    fig.suptitle(f"{dataset_name}: Confusion Matrices", fontsize=17, y=1.02)
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_confusion_matrices.png")


def plot_model_comparison(results, metric):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=results, x="Dataset", y=metric, hue="Model",
        palette={"RBF-SVM": "#4C78A8", "Random Forest": "#F58518"}
    )
    ax.set_title(f"Model Comparison: {metric}")
    lower = max(0, results[metric].min() - 0.08)
    upper = min(1.02, results[metric].max() + 0.05)
    ax.set_ylim(lower, upper)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    save_and_show_figure(f"model_comparison_{metric.lower().replace(' ', '_')}.png")


def plot_performance_crossover(results, metric="Test Accuracy"):
    pivot = results.pivot(index="Dataset", columns="Model", values=metric).reindex(["Rice", "Dry Bean"])
    plt.figure(figsize=(10, 6))
    colors = {"RBF-SVM": "#4C78A8", "Random Forest": "#F58518"}
    for model_name in pivot.columns:
        plt.plot(pivot.index, pivot[model_name], marker="o", markersize=10,
                 linewidth=2.8, label=model_name, color=colors[model_name])
        for dataset_name, value in pivot[model_name].items():
            plt.annotate(f"{value:.3f}", (dataset_name, value), xytext=(0, 8),
                         textcoords="offset points", ha="center")
    plt.title(f"Model Performance Across Datasets: {metric}")
    plt.xlabel("Dataset")
    plt.ylabel(metric)
    plt.legend(title="Model")
    save_and_show_figure(f"performance_crossover_{metric.lower().replace(' ', '_')}.png")


def evaluate_cv_stability(X, y, fitted_models, dataset_name):
    y_encoded = LabelEncoder().fit_transform(y.astype(str))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    records = []
    for model_name, result in fitted_models.items():
        scores = cross_validate(
            result["model"], X, y_encoded, cv=cv,
            scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"},
            n_jobs=-1
        )
        for i in range(len(scores["test_accuracy"])):
            records.append({
                "Dataset": dataset_name, "Model": model_name, "Split": i + 1,
                "Accuracy": scores["test_accuracy"][i],
                "Balanced Accuracy": scores["test_balanced_accuracy"][i],
                "Macro F1": scores["test_macro_f1"][i],
            })
    return pd.DataFrame(records)


def plot_cv_boxplot(cv_results, metric):
    plt.figure(figsize=(11, 7))
    sns.boxplot(data=cv_results, x="Dataset", y=metric, hue="Model",
                palette={"RBF-SVM": "#4C78A8", "Random Forest": "#F58518"})
    sns.stripplot(data=cv_results, x="Dataset", y=metric, hue="Model",
                  dodge=True, color="black", alpha=0.30, size=3, legend=False)
    plt.title(f"Repeated Cross-validation Distribution: {metric}")
    save_and_show_figure(f"cv_distribution_{metric.lower().replace(' ', '_')}.png")


def plot_learning_curve_for_model(model, X, y, model_name, dataset_name):
    y_encoded = LabelEncoder().fit_transform(y.astype(str))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sizes, train_scores, val_scores = learning_curve(
        model, X, y_encoded, train_sizes=np.linspace(0.15, 1.0, 6), cv=cv,
        scoring="accuracy", n_jobs=-1, shuffle=True, random_state=RANDOM_STATE
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, train_mean, marker="o", label="Training Accuracy", color="#4C78A8")
    plt.plot(sizes, val_mean, marker="s", label="Validation Accuracy", color="#F58518")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.18, color="#4C78A8")
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.18, color="#F58518")
    plt.title(f"{dataset_name}: {model_name} Learning Curve")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    safe = f"{dataset_name}_{model_name}".lower().replace(" ", "_").replace("-", "_")
    save_and_show_figure(f"{safe}_learning_curve.png")


def plot_svm_parameter_heatmap(fitted_models, dataset_name):
    data = pd.DataFrame(fitted_models["RBF-SVM"]["grid_search"].cv_results_)
    data["C"] = data["param_model__C"].astype(str)
    data["Gamma"] = data["param_model__gamma"].astype(str)
    heat = data.pivot_table(index="Gamma", columns="C", values="mean_test_score", aggfunc="mean")
    plt.figure(figsize=(9, 6))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "Mean CV Accuracy"})
    plt.title(f"{dataset_name}: RBF-SVM Parameter Performance")
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_svm_parameter_heatmap.png")


def plot_rf_feature_importance(fitted_models, X, dataset_name, top_n=12):
    rf = fitted_models["Random Forest"]["model"].named_steps["model"]
    data = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
    data = data.nlargest(top_n, "Importance").sort_values("Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(data["Feature"], data["Importance"], color="#F58518")
    plt.title(f"{dataset_name}: Random Forest Feature Importance")
    plt.xlabel("Importance")
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_rf_feature_importance.png")


def plot_permutation_importance(fitted_models, X, dataset_name, model_name, top_n=12):
    result = fitted_models[model_name]
    importance = permutation_importance(
        result["model"], result["X_test"], result["y_test"], scoring="accuracy",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    data = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance.importances_mean,
        "Std": importance.importances_std,
    }).nlargest(top_n, "Importance").sort_values("Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(data["Feature"], data["Importance"], xerr=data["Std"], color="#54A24B", alpha=0.85)
    plt.title(f"{dataset_name}: {model_name} Permutation Importance")
    plt.xlabel("Decrease in Test Accuracy after Permutation")
    safe = f"{dataset_name}_{model_name}".lower().replace(" ", "_").replace("-", "_")
    save_and_show_figure(f"{safe}_permutation_importance.png")


def calculate_class_metrics(fitted_models, dataset_name):
    records = []
    for model_name, result in fitted_models.items():
        report = classification_report(
            result["y_test"], result["y_pred"], target_names=result["label_encoder"].classes_,
            output_dict=True, zero_division=0
        )
        for class_name in result["label_encoder"].classes_:
            records.append({
                "Dataset": dataset_name, "Model": model_name, "Class": class_name,
                "Precision": report[class_name]["precision"],
                "Recall": report[class_name]["recall"],
                "F1-score": report[class_name]["f1-score"],
                "Support": report[class_name]["support"],
            })
    return pd.DataFrame(records)


def plot_class_recall(class_metrics, dataset_name):
    data = class_metrics[class_metrics["Dataset"] == dataset_name]
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=data, x="Class", y="Recall", hue="Model",
                     palette={"RBF-SVM": "#4C78A8", "Random Forest": "#F58518"})
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title(f"{dataset_name}: Recall by Variety")
    save_and_show_figure(f"{dataset_name.lower().replace(' ', '_')}_recall_by_variety.png")


def analyse_ranking_reversal(results, metric="Test Accuracy"):
    """输出两个数据集各自的最佳模型，并判断是否发生排名反转。"""
    winners = {}
    print("\n" + "=" * 70)
    print(f"排名反转分析：{metric}")
    for dataset_name in results["Dataset"].unique():
        subset = results[results["Dataset"] == dataset_name]
        best = subset.loc[subset[metric].idxmax()]
        winners[dataset_name] = best["Model"]
        print(f"{dataset_name}: {best['Model']}，{metric}={best[metric]:.4f}")
    if len(set(winners.values())) > 1:
        print("结论：不同数据集的最佳模型不同，出现模型排名反转。")
    else:
        print("结论：未出现严格排名反转；应继续比较性能差距、稳定性和类别级 Recall。")


# ============================== 主程序 ==============================
def main():
    # 1. 下载并清理数据
    X_rice, y_rice = load_uci_dataset(545, "Rice")
    X_bean, y_bean = load_uci_dataset(602, "Dry Bean")
    X_rice, y_rice = clean_dataset(X_rice, y_rice, "Rice")
    X_bean, y_bean = clean_dataset(X_bean, y_bean, "Dry Bean")

    # 2. 数据质量与探索性可视化
    for X, y, name in [(X_rice, y_rice, "Rice"), (X_bean, y_bean, "Dry Bean")]:
        inspect_dataset(X, y, name)
        plot_class_distribution(y, name)
        plot_feature_histograms(X, name)
        plot_correlation_heatmap(X, name)
        plot_pca_projection(X, y, name)

    # 3. 训练两个模型
    configurations = get_model_configurations(FULL_GRID_SEARCH)
    rice_results, rice_models = train_and_evaluate_models(X_rice, y_rice, "Rice", configurations)
    bean_results, bean_models = train_and_evaluate_models(X_bean, y_bean, "Dry Bean", configurations)
    all_results = pd.concat([rice_results, bean_results], ignore_index=True)
    all_results.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)
    print("\n综合结果：")
    print(all_results.round(4).to_string(index=False))

    # 4. 模型结果可视化
    plot_confusion_matrices(rice_models, "Rice")
    plot_confusion_matrices(bean_models, "Dry Bean")
    for metric in ["Test Accuracy", "Balanced Accuracy", "Macro F1"]:
        plot_model_comparison(all_results, metric)
    plot_performance_crossover(all_results, "Test Accuracy")
    plot_performance_crossover(all_results, "Macro F1")

    # 5. 重复交叉验证稳定性
    rice_cv = evaluate_cv_stability(X_rice, y_rice, rice_models, "Rice")
    bean_cv = evaluate_cv_stability(X_bean, y_bean, bean_models, "Dry Bean")
    all_cv = pd.concat([rice_cv, bean_cv], ignore_index=True)
    all_cv.to_csv(os.path.join(OUTPUT_DIR, "repeated_cross_validation_results.csv"), index=False)
    plot_cv_boxplot(all_cv, "Accuracy")
    plot_cv_boxplot(all_cv, "Macro F1")
    cv_summary = all_cv.groupby(["Dataset", "Model"])[["Accuracy", "Balanced Accuracy", "Macro F1"]].agg(["mean", "std"])
    cv_summary.to_csv(os.path.join(OUTPUT_DIR, "cross_validation_summary.csv"))

    # 6. 学习曲线与参数分析
    for model_name, result in rice_models.items():
        plot_learning_curve_for_model(result["model"], X_rice, y_rice, model_name, "Rice")
    for model_name, result in bean_models.items():
        plot_learning_curve_for_model(result["model"], X_bean, y_bean, model_name, "Dry Bean")
    plot_svm_parameter_heatmap(rice_models, "Rice")
    plot_svm_parameter_heatmap(bean_models, "Dry Bean")

    # 7. 特征重要性
    plot_rf_feature_importance(rice_models, X_rice, "Rice", top_n=7)
    plot_rf_feature_importance(bean_models, X_bean, "Dry Bean", top_n=12)
    for model_name in ["RBF-SVM", "Random Forest"]:
        plot_permutation_importance(rice_models, X_rice, "Rice", model_name, top_n=7)
        plot_permutation_importance(bean_models, X_bean, "Dry Bean", model_name, top_n=12)

    # 8. 类别级结果
    class_metrics = pd.concat([
        calculate_class_metrics(rice_models, "Rice"),
        calculate_class_metrics(bean_models, "Dry Bean"),
    ], ignore_index=True)
    class_metrics.to_csv(os.path.join(OUTPUT_DIR, "class_level_metrics.csv"), index=False)
    plot_class_recall(class_metrics, "Rice")
    plot_class_recall(class_metrics, "Dry Bean")

    # 9. 判断是否发生排名反转
    analyse_ranking_reversal(all_results, "Test Accuracy")
    analyse_ranking_reversal(all_results, "Macro F1")
    print(f"\n全部结果已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
