import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel

print("============================================================")
print("🚀 终极外科手术：制造免重建的 4通道 Pose 预训练模型")
print("============================================================")

# 1. 加载官方老戏骨
print("\n1. 加载 yolov8n-pose.pt...")
old_model = YOLO('yolov8n-pose.pt')
old_state_dict = old_model.model.state_dict()

# 2. 提取并修改配置 (骗过 YOLO 的安检系统)
print("2. 正在修改网络基因图纸 (强制 ch=4, nc=1, kpt_shape=[3, 3])...")
yaml_cfg = old_model.model.yaml.copy()
yaml_cfg['ch'] = 4
yaml_cfg['nc'] = 1             # 类别数改成 1 (abdomen)
yaml_cfg['kpt_shape'] = [3, 3] # 关键点改成 3 个 (肚脐 + 2个穴位)

# 3. 按照新图纸构建全新模型骨架
print("3. 正在构建全新维度的模型骨架...")
new_model = PoseModel(cfg=yaml_cfg, ch=4, nc=1, data_kpt_shape=[3, 3])
new_state_dict = new_model.state_dict()

# 4. 移植权重
print("4. 开始精准移植老戏骨权重...")
for k, v in new_state_dict.items():
    if k in old_state_dict and v.shape == old_state_dict[k].shape:
        # 形状完全匹配的深层网络，直接复制过来
        new_state_dict[k] = old_state_dict[k]
    elif k == 'model.0.conv.weight':
        # 第一层卷积：3通道变4通道的微创手术
        print(f"   >>> 正在对 {k} 进行 4 通道扩容手术...")
        new_state_dict[k][:, :3, :, :] = old_state_dict[k]
        torch.nn.init.normal_(new_state_dict[k][:, 3:, :, :], mean=0, std=0.01)
    else:
        # 头部预测层（因为从17个点变成了3个点，形状不一样了，必须保留新骨架的随机初始化）
        pass

new_model.load_state_dict(new_state_dict)

# 5. 保存完美匹配的预训练模型
ckpt = {
    'model': new_model.half(),
    'train_args': {},
    'epoch': -1
}
# 取个新名字防止混淆
torch.save(ckpt, 'yolov8n-pose_4ch_perfect.pt')
print("\n✅ 手术圆满成功！已生成: yolov8n-pose_4ch_perfect.pt")
print("现在 YOLO 引擎会完美识别它，不会再偷偷重置通道了！")