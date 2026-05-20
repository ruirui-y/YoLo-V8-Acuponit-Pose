import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel

print("============================================================")
print("🚀 终极外科手术 3.0：最稳的 4通道 Pose 模型")
print("============================================================")

print("\n1. 加载 yolov8n-pose.pt...")
old_model = YOLO('yolov8n-pose.pt')
old_state_dict = old_model.model.state_dict()

print("2. 正在修改网络图纸...")
yaml_cfg = old_model.model.yaml.copy()
yaml_cfg['ch'] = 4
yaml_cfg['nc'] = 1
yaml_cfg['kpt_shape'] = [3, 3]

print("3. 构建全新维度的模型骨架...")
# 【优化1】不用 half()，全精度训练更稳，避免4通道数值溢出
new_model = PoseModel(cfg=yaml_cfg, ch=4, nc=1, data_kpt_shape=[3, 3])
new_state_dict = new_model.state_dict()

print("4. 开始移植老戏骨权重...")
for k, v in new_state_dict.items():
    if k in old_state_dict and v.shape == old_state_dict[k].shape:
        new_state_dict[k] = old_state_dict[k]
    elif k == 'model.0.conv.weight':
        # 【优化2】第4通道不用随机初始化，用RGB的平均值，更稳定
        new_state_dict[k][:, :3, :, :] = old_state_dict[k]
        rgb_mean = old_state_dict[k].mean(dim=1, keepdim=True)
        new_state_dict[k][:, 3:, :, :] = rgb_mean
        print(f"   ✅ 第一层卷积：前3通道用官方，第4通道用RGB平均初始化")

new_model.load_state_dict(new_state_dict)
new_model.names = {0: 'abdomen'}

# 5. 🔑 终极杀招：把“身份证”一起存进去！
ckpt = {
    'model': new_model,  # 【优化3】去掉 .half()，全精度保存
    'train_args': {
        'task': 'pose',
        'mode': 'train',
        'epochs': 300,
        'batch': 4,
        'imgsz': 640
    },  # 补全基本训练参数
    'epoch': -1,
    'task': 'pose',
    'names': {0: 'abdomen'},
    'nc': 1,
    'kpt_shape': [3, 3]
}

save_path = 'yolov8n-pose_4ch_perfect.pt'
torch.save(ckpt, save_path)
print(f"\n✅ 手术圆满成功！已生成: {save_path}")
print("这回它带着合法身份证，YOLO 引擎绝对不会再认错了！")