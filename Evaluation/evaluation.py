import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from PIL import Image
import os

# 读取图像并转换为数组
def load_image_as_array(file_path):
    with Image.open(file_path) as img:
        return np.array(img)

# 将 255 像素值转换为 1（二值化）
def convert_to_binary(img_array):
    return (img_array == 255).astype(int)

# 文件夹路径

predictions_folder = '...'
ground_truths_folder = '...'

# 获取文件名列表
prediction_files = os.listdir(predictions_folder)
ground_truth_files = os.listdir(ground_truths_folder)

# 创建 ground truth 映射
ground_truth_dict = {}
for gt_file in ground_truth_files:
    date_time_part = gt_file.split('.')[-2]
    ground_truth_dict[date_time_part] = gt_file

# 存储评估结果
evaluation_results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'iou': [],
    'dice': []
}

# 逐张图像进行评估
for pred_file in prediction_files:
    date_time_part = pred_file.split('.')[-2]
    if date_time_part in ground_truth_dict:
        pred_path = os.path.join(predictions_folder, pred_file)
        gt_path = os.path.join(ground_truths_folder, ground_truth_dict[date_time_part])

        pred = convert_to_binary(load_image_as_array(pred_path)).flatten()
        gt = convert_to_binary(load_image_as_array(gt_path)).flatten()

        # 检查 pred 和 gt 是否全零
        pred_sum = np.sum(pred)
        gt_sum = np.sum(gt)
        print(f"Processing {pred_file}: Pred sum = {pred_sum}, GT sum = {gt_sum}")

        # 计算指标（避免 zero_division 警告）
        evaluation_results['accuracy'].append(accuracy_score(gt, pred))
        evaluation_results['precision'].append(precision_score(gt, pred, zero_division=1))
        evaluation_results['recall'].append(recall_score(gt, pred, zero_division=1))
        evaluation_results['f1_score'].append(f1_score(gt, pred, zero_division=1))
        evaluation_results['iou'].append(jaccard_score(gt, pred, zero_division=1))

        # 计算 Dice 系数
        if pred_sum + gt_sum == 0:
            dice_score = 1.0
        else:
            dice_score = 2 * np.sum(pred * gt) / (pred_sum + gt_sum)
        evaluation_results['dice'].append(dice_score)

# 计算所有指标的均值
for key in evaluation_results:
    evaluation_results[key] = np.mean(evaluation_results[key])

# 输出评估结果
print("\nEvaluation Results:")
for key, value in evaluation_results.items():
    print(f"{key.capitalize()}: {value:.4f}")