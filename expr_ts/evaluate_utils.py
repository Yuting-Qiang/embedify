# from datetime import datetime

# import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

# import numpy as np

# from data.decision_tree import construct_dataset

# TEST_START_DATE = datetime.strptime(config.simulate_start_date, "%Y-%m-%d")
# TEST_END_DATE = datetime.strptime(config.simulate_end_date, "%Y-%m-%d")


def precision_threshold_recall(precision, threshold, recall, require_precision):
    k = sum(precision > require_precision)
    return threshold[len(threshold) - k], recall[len(recall) - k]


def plot_precision_threshold_recall(scores, labels, model_name):
    # 计算正例和负例的精度-召回率曲线
    precision_pos, recall_pos, thresholds_pos = precision_recall_curve(labels == 1, scores)
    precision_neg, recall_neg, thresholds_neg = precision_recall_curve(labels == 0, 1 - scores)
    precision_threshold_recall_dict = {}
    th, re = precision_threshold_recall(
        precision=precision_pos, threshold=thresholds_pos, recall=recall_pos, require_precision=0.9
    )
    precision_threshold_recall_dict["p90_th"] = th
    precision_threshold_recall_dict["p90_re"] = re
    th, re = precision_threshold_recall(
        precision=precision_pos, threshold=thresholds_pos, recall=recall_pos, require_precision=0.8
    )
    precision_threshold_recall_dict["p80_th"] = th
    precision_threshold_recall_dict["p80_re"] = re
    th, re = precision_threshold_recall(
        precision=precision_pos, threshold=thresholds_pos, recall=recall_pos, require_precision=0.7
    )
    precision_threshold_recall_dict["p70_th"] = th
    precision_threshold_recall_dict["p70_re"] = re
    th, re = precision_threshold_recall(
        precision=precision_pos, threshold=thresholds_pos, recall=recall_pos, require_precision=0.6
    )
    precision_threshold_recall_dict["p60_th"] = th
    precision_threshold_recall_dict["p60_re"] = re
    th, re = precision_threshold_recall(
        precision=precision_pos, threshold=thresholds_pos, recall=recall_pos, require_precision=0.5
    )
    precision_threshold_recall_dict["p50_th"] = th
    precision_threshold_recall_dict["p50_re"] = re
    print(precision_threshold_recall_dict)
    plt.figure(figsize=(8, 6))

    # 绘制正例曲线
    plt.plot(recall_pos, precision_pos, label="Positive", color="blue")

    # 绘制负例曲线
    plt.plot(recall_neg, precision_neg, label="Negative", color="red")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Threshold-Recall Curve")
    plt.legend()
    plt.savefig(f"{model_name}_pr_curve.png")
    plt.show()
    return precision_threshold_recall_dict


def plot_roc_curve(predictions, labels, model_name):
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"{model_name}_roc_curve.png")
    plt.show()
    return auc


# def test(config):
#     # 加载模型
#     bst = lgb.Booster(model_file=config.model_path)  # init model

#     # 准备测试数据集
#     test_dataset = construct_dataset(
#         datetime.strptime(config.simulate_start_date, "%Y-%m-%d"),
#         datetime.strptime(config.simulate_end_date, "%Y-%m-%d"),
#         pre_days=config.pre_days,
#         post_days=config.post_days,
#     )

#     test_dataset.dropna(inplace=True)
#     for i in range(config.pre_days):
#         test_dataset = test_dataset[
#             (test_dataset[f"increase_p{i+1}p{i}"] < 0.5)
#             & (test_dataset[f"increase_p{i+1}p{i}"] > -0.5)
#         ]
#     test_dataset.to_csv(
#         f"test_dataset_{config.simulate_start_date}_{config.simulate_end_date}.csv"
#     )
#     # import pandas as pd
#     # test_dataset = pd.read_csv(f"test_dataset_{config.simulate_start_date}_{config.simulate_end_date}.csv")
#     # test_dataset = pd.read_csv(f"test_dataset_2025-01-01_2025-01-31.csv")
#     # 预测结果、计算指标
#     predictions = bst.predict(
#         test_dataset[config.feature_cols]
#         # (test_dataset[config.feature_cols] - test_dataset[config.feature_cols].mean())/test_dataset[config.feature_cols].std()
#     )
#     labels = np.max(
#         np.stack([test_dataset[f"High_n{i+1}"].to_numpy() for i in range(config.post_days)]),
#         axis=0,
#     ) > (test_dataset.Close_p0.to_numpy() * 1.10)
#     metrics = {"model_name": config.model_name}
#     # exception_threshold = sorted(predictions)[-10]
#     # selected_index = predictions < exception_threshold
#     # predictions = predictions[selected_index]
#     # labels = labels[selected_index]
#     auc = plot_roc_curve(predictions, labels)
#     metrics["auc"] = auc
#     precision_threshold_recall_dict = plot_precision_threshold_recall(predictions, labels)
#     metrics.update(precision_threshold_recall_dict)
#     metrics["f1_score"] = f1_score(labels, predictions > metrics["p50_th"])
#     print(metrics["f1_score"])
#     return metrics
