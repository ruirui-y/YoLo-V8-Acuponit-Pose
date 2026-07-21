"""验证 Ultralytics 数据加载器是否读取 4 通道图片"""
from ultralytics.data import build_dataset
import torch

# 用你的 YAML 构建数据集
dataset = build_dataset(
    cfg='datasets/acupoint.yaml',   # 换成你实际用的 YAML
    batch=4,
    mode='train',
    rect=False,
    stride=32
)

# 加载第一张图
img = dataset[0]['img']
print(f"加载后的图片形状: {img.shape}")  # 期待: [3, 640, 640] 或 [4, 640, 640]
print(f"通道数: {img.shape[0]}")

if img.shape[0] == 3:
    print("❌ 数据加载器只读了 3 通道！深度通道被丢弃了")
elif img.shape[0] == 4:
    print("✅ 数据加载器正确读取了 4 通道！")
else:
    print(f"⚠️ 未知通道数: {img.shape[0]}")