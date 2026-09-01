"""
将 3 通道 Pose 权重转换为 4 通道 RGBD Pose 权重。

转换逻辑（已验证）：
- 第 1 层卷积的前 3 个输入通道（RGB）权重原样保留
- 第 4 个输入通道（Depth）使用 RGB 三通道卷积权重的均值初始化
- 其余可匹配的权重保持原样
- 模型其余结构、类别数(nc)、关键点维度(kpt_shape)、类别名(names)
  均沿用输入 3 通道权重；输入通道维度改为 4

关键点结构不在本脚本写死：kpt_shape/nc/names 由 --input 权重决定，
后续在训练阶段由 data.yaml 覆盖（如 hand 实验的 kpt_shape=[7,3]）。
"""
import argparse
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel


def parse_args():
    p = argparse.ArgumentParser(description="3通道 Pose 权重 -> 4通道 RGBD Pose 权重")
    p.add_argument('--input', required=True, help='输入 3 通道 Pose 权重 .pt 路径')
    p.add_argument('--output', required=True, help='输出 4 通道 RGBD Pose 权重 .pt 路径')
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("3通道 Pose 权重 -> 4通道 RGBD Pose 权重")
    print("=" * 60)

    # 1. 加载输入 3 通道权重
    print(f"\n1. 加载输入权重: {args.input}")
    old_model = YOLO(args.input, task='pose')
    old_state_dict = old_model.model.state_dict()
    old_yaml = old_model.model.yaml

    # 2. 修改网络图纸：仅把输入通道改为 4，其余沿用输入权重
    print("2. 修改网络图纸 (ch=4)...")
    yaml_cfg = old_yaml.copy()
    yaml_cfg['ch'] = 4
    nc = yaml_cfg.get('nc', 1)
    kpt_shape = yaml_cfg.get('kpt_shape', [17, 3])
    names = old_model.names if hasattr(old_model, 'names') else {0: 'object'}

    # 3. 构建 4 通道模型骨架（关键点维度/类别沿用输入权重）
    print("3. 构建 4 通道模型骨架...")
    new_model = PoseModel(cfg=yaml_cfg, ch=4, nc=nc, data_kpt_shape=kpt_shape)
    new_state_dict = new_model.state_dict()

    # 4. 移植权重
    print("4. 移植权重...")
    for k, v in new_state_dict.items():
        if k in old_state_dict and v.shape == old_state_dict[k].shape:
            new_state_dict[k] = old_state_dict[k]
        elif k == 'model.0.conv.weight':
            # 第4通道不用随机初始化，用 RGB 三通道卷积权重的均值，更稳定
            new_state_dict[k][:, :3, :, :] = old_state_dict[k]
            rgb_mean = old_state_dict[k].mean(dim=1, keepdim=True)
            new_state_dict[k][:, 3:, :, :] = rgb_mean
            print("   第一层卷积：前3通道原样，第4通道用 RGB 均值初始化")

    new_model.load_state_dict(new_state_dict)

    # 5. 检查第一层输入通道 == 4
    first_w = new_model.state_dict()['model.0.conv.weight']
    in_ch = first_w.shape[1]
    print(f"\n5. 第一层卷积输入通道数: {in_ch}")
    assert in_ch == 4, f"❌ 第一层输入通道应为 4，实际为 {in_ch}"
    print("   ✅ 第一层输入通道 == 4 校验通过")

    # 6. 保存（kpt_shape/nc/names 沿用输入权重；train_args 不再写死实验超参，
    #    真实训练参数由训练脚本/ data.yaml 决定）
    ckpt = {
        'model': new_model,
        'train_args': {
            'task': 'pose',
            'mode': 'train',
        },
        'epoch': -1,
        'task': 'pose',
        'names': names,
        'nc': nc,
        'kpt_shape': kpt_shape,
    }
    torch.save(ckpt, args.output)
    print(f"\n✅ 已生成 4 通道权重: {args.output}")


if __name__ == '__main__':
    main()
