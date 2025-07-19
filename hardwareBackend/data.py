import torch
import pandas as pd
import numpy as np

def normalize_keypoints(xy_array):
    keypoints = xy_array.reshape(-1, 2)  # (21, 2)
    center = keypoints[0]                # 一般以手腕点为中心
    keypoints -= center                  # 平移到原点

    max_dist = np.max(np.linalg.norm(keypoints, axis=1)) + 1e-6
    keypoints /= max_dist                # 尺度归一化

    return keypoints.flatten()           # 展平为 (42,)

def read_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        coords = row[:-1].values.astype(np.float32)
        label = row["label"]
        coords_norm = normalize_keypoints(coords)
        X.append(coords_norm)
        y.append(label)
    return np.array(X), np.array(y)

def augment_keypoints(xy_array):
    keypoints = xy_array.reshape(-1, 2)

    # 随机镜像
    if np.random.rand() < 0.5:
        keypoints[:, 0] *= -1

    # 添加小扰动
    noise = np.random.normal(0, 0.01, keypoints.shape)
    keypoints += noise

    # 随机缩放
    scale = np.random.uniform(0.9, 1.1)
    keypoints *= scale

    return keypoints.flatten()

def rotate_keypoints(keypoints, angle_deg):
    angle = np.radians(angle_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return (keypoints @ R.T)


def save_to_tensor_file(X, y, out_path="keypoints_dataset.pt"):
    label_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, 'OK': 6, 'X': 7}
    y_indices = np.array([label_map[label] for label in y], dtype=np.int64)

    data = {
        'X': torch.tensor(X).float(),  # shape: (N, 42)
        'y': torch.tensor(y_indices).long()  # shape: (N,)
    }
    torch.save(data, out_path)
    print(f"数据保存到 {out_path}，共样本数: {len(X)}")

def augment_dataset(X, y):
    """对整个数据集进行增强：原样 + 增强后的版本"""
    X_aug, y_aug = [], []
    angles = [-50, -20, 10, 20]
    for xi, yi in zip(X, y):
        X_aug.append(xi)
        y_aug.append(yi)

        keypoints = xi.reshape(-1, 2)

        # 镜像样本
        mirrored = keypoints.copy()
        mirrored[:, 0] *= -1
        X_aug.append(mirrored.flatten())
        y_aug.append(yi)

        # 多角度旋转增强
        for angle in angles:
            rotated = rotate_keypoints(keypoints, angle)
            X_aug.append(rotated.flatten())
            y_aug.append(yi)

            # 旋转 + 镜像
            rotated_mirror = rotated.copy()
            rotated_mirror[:, 0] *= -1
            X_aug.append(rotated_mirror.flatten())
            y_aug.append(yi)

        # 加入原来的镜像+噪声增强
        aug = augment_keypoints(xi)
        X_aug.append(aug)
        y_aug.append(yi)

    return np.array(X_aug), np.array(y_aug)

def full_preprocess_pipeline(csv_path, out_path="keypoints_dataset.pt"):
    X, y = read_and_preprocess(csv_path)         # 读取并归一化原始数据
    X_aug, y_aug = augment_dataset(X, y)         # 原始 + 增强版本
    save_to_tensor_file(X_aug, y_aug, out_path)  # 保存为 .pt

def main():
    full_preprocess_pipeline("keypoints_val.csv", "keypoints_val.pt")

if __name__ == "__main__":
    main()
