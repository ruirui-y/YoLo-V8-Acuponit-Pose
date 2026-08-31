直接拉取项目[https://github.com/Zhou9687/yolov8-rgbd-detection.git](https://github.com/Zhou9687/yolov8-rgbd-detection.git)



# 配置环境
```cpp
git clone https://github.com/m-sir-zhou/yolov8-rgbd-detection.git
cd yolov8-rgbd-detection
```



```cpp
# 激活conda环境
conda activate yolov8

# 配置yolov8环境
# 安装带GPU加速的PyTorch 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装yolov8相关依赖
pip install -e .

# 验证CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```



创建4通道权重， 运行成功后，你的根目录下会多出一个名为 `yolov8_4ch_direct.pt` 的权重文件，这就是带着丰富视觉经验的 4 通道初始模型  

```cpp
python scripts/prepare_4ch_weights.py
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774857113836-97770317-92a3-408b-a5d8-6d4b52e99ed0.png)



## 配置数据集
素材需要一张RGBD的图片

目前手头上的图片都是分开的，需要进行融合

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774857510002-f3c7639a-6689-4ce4-9530-5f39281a8695.png)



融合训练集

```cpp
python scripts/fuse_rgb_depth.py --rgb_dir "H:\YJJ\YOLO_Project\rgb_dataset\train\images" --depth_dir "H:\YJJ\YOLO_Project\rgb_dataset\train\depth" --out_dir "datasets/acupoint/images/train" --depth_type uint8 --mode name
```



融合验证集

```cpp
python scripts/fuse_rgb_depth.py --rgb_dir "H:\YJJ\YOLO_Project\rgb_dataset\val\images" --depth_dir "H:\YJJ\YOLO_Project\rgb_dataset\val\depth" --out_dir "datasets/acupoint/images/val" --depth_type uint8 --mode name
```



训练完毕之后，创建yaml

```cpp
# 1. 数据集绝对路径
path: H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint
train: images/train
val: images/val

# 2. 类别配置 (穴位检测为 1 类)
nc: 1
names:
  0: acupoint

# 3. 🔑 核心 RGBD 魔改配置 (必填)
rgbd: true
channels: 4
```



文件夹视图

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774858332395-bdadb17a-9a8c-48a8-a35c-412011db1777.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774858344238-4706bc9f-c59e-4326-a3f0-6211ff74d2eb.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774858358130-55ba636d-85a8-44e1-8cdb-aaa4194f0e18.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774858379327-14545d8f-31da-4997-8456-547c308b8a42.png)



## 修改启动脚本
 打开项目根目录下的 `train_rgbd_direct.py` 文件  

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774858479415-d17fefdb-a5f7-47e7-a702-55d34f651cba.png)



换成自己的yaml路径

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774858533926-c84aa1c2-8122-4371-af82-265748512a6f.png)



## 开始训练
```cpp
python train_rgbd_direct.py
```



## 穴位尺寸过小导致box_loss无穷大
通过扩大标签的尺寸，解决了这个问题

```python
import os
import glob

# 你的标签文件夹路径 (请确认路径对不对)
label_dirs = [
    r"datasets\acupoint\labels\train",
    r"datasets\acupoint\labels\val"
]

NEW_SIZE = "0.050000"  # 把宽高统一放大到 5%

for d in label_dirs:
    if not os.path.exists(d):
        print(f"找不到文件夹: {d}")
        continue

    txt_files = glob.glob(os.path.join(d, "*.txt"))
    count = 0

    for txt_path in txt_files:
        with open(txt_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # 保持 class, x_center, y_center 不变，强行替换 width 和 height
                cls_id, x, y, _, _ = parts
                new_line = f"{cls_id} {x} {y} {NEW_SIZE} {NEW_SIZE}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        with open(txt_path, "w") as f:
            f.writelines(new_lines)

        count += 1
    print(f"成功放大了 {d} 下的 {count} 个标签文件！")

print("全部处理完成，可以重新开始训练了！")
```



##  mAP 参数过小
目标太小了，数据太少了

调整参数，加大轮次，继续

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1774950071805-ccc9fdbc-4338-41d4-a8aa-2b665a05bf6c.png)





### 测试
```cpp
python predict_rgbd.py
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775124892404-93e77d0a-5d07-4d38-9a48-19cac7d8d67c.png)



## 反思
穴位的像素和别的皮肤特征没有任何区别，我在想如果把穴位的识别变成肚脐的识别，然后按照肚脐的位置反过来去推演每一个穴位的位置呢？

但是这种只对一个穴位有效，如果是3个穴位要依赖什么特征呢？

如果是提前标注上3个穴位的相对位置，我用一个程序强行归一化这三个穴位的相对位置，在yolo里面只负责去识别肚脐的位置，推算出第一个穴位的位置，然后反过来去推算另外两个



**计划**

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775185172074-7b296153-77ed-40f1-bfbd-26035e75dbcd.png)



## yolo训练的本质
yolo是在训练什么呢？他是怎么根据我给的数据进行训练的，标签又在对标什么？



一个盲人学生在做海量的看图说话练习册

yolo内部是由几百万甚至几千万个全中参数组成的数学矩阵，刚开始训练时，这些权重是完全随机的数字，训练的本质是通过不断看图，微调这几百万个数字，训练的结果，这些数字会被调整到一个完美的组合



yolo在看图片是，会把640*640的图片强行分为一张网格(Grid)，标签的作用就是告诉yolo目标在哪一个网格里面



训练过程

yolo拿起一张图，由于权重是随机的，会随便画上1000个框，然后去和标签对答案，box loss就是画的框和真实标签标注的框，中心点差了多少像素点，los loss是分类损失，yolo的法官会根据这些差距得出一个总分，loss越大，说明错的越离谱，然后模型再去对权重进行微调，学习率lr0，就是每次修改的幅度



标签：

0是类别，也就是穴位，0.55中心点x坐标(位于整张图片宽度从左往右55.17%的位置)

0.38中心点的Y坐标( 穴位中心点位于图片高度从上往下 **38.55%** 的位置  )

0.05，0.05是框的宽度和高度

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775186065636-65aa4c57-c4da-4f4e-9e55-9a591f78ef0e.png)



## 修改模型为pose
原作者已经将“多模态数据输入，四通道卷积融合，放报错机制”全部踩坑解决了

所以这里我们只需要下载官方的Pose预训练模型，然后装上第四个通道，就好了

```python
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
```



### 数据集打标签
1. 把整个腹部(或者包含肚脐的明显区域)框起来
2. 标上关键点：在框内打上需要定位的坐标明
3. Pose标签的新格式

 类别ID 大框中心X 大框中心Y 大框宽 大框高 点1_X 点1_Y 可见度 点2_X 点2_Y 可见度  

```sql
0 0.445313 0.472917 0.332813 0.226389 0.411681 0.479899 2 0.482906 0.476100 2 0.532764 0.467236 2
```



### 程序化生成
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775635889748-cd694982-1604-45d4-a293-ff6c36fd6509.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775635928267-f7ac4c56-b2c0-4998-a1cc-e184edebf8ea.png)



### 重新配置数据集
首先将程序生成的新标签，全部替换到指定的文件夹

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775635954224-a5a38e98-fc2a-452e-9176-dffde5271e5c.png)



然后配置yaml文件

```yaml
# 1. 数据集绝对路径 (非常好，继续保持)
path: H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint
train: images/train
val: images/val

# 2. 类别配置 (正确，大框是腹部)
nc: 1
names:
  0: abdomen

# 3. RGBD 配置 (如果你的代码里有读取这两个参数的逻辑就留着，没有可以删掉)
# 注意：如果是手动修改的数据集加载代码，这两个参数可能没用，以代码修改为准
rgbd: true
channels: 4

# 4. 🎯 Pose 模型核心配置 (你之前写对了)
kpt_shape: [3, 3]  # 3个关键点，每个点(x,y,visibility)

# 5. 🔑 必须补的！之前报错就是因为没加这个
# 翻转索引：假设你的3个点是 [左穴位, 肚脐, 右穴位]，左右翻转时左↔右互换，肚脐不变
flip_idx: [2, 1, 0]
```



### 训练前测试
```python
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


def test_simple():
    print("=" * 60)
    print("🧪 简化版测试：2分钟确认核心环节")
    print("=" * 60)

    # ================== 1. 测试模型 ==================
    print("\n[1/3] 测试模型...")
    model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')
    first_layer = model.model.model[0].conv
    in_channels = first_layer.weight.shape[1]
    print(f"   模型输入通道数: {in_channels}")
    assert in_channels == 4, "❌ 模型不是4通道！"
    print("   ✅ 模型验证通过")

    # ================== 2. 测试图片和标签 ==================
    print("\n[2/3] 测试图片和标签...")
    # 找一张测试图和对应的标签
    img_dir = Path('datasets/acupoint/images/val')
    label_dir = Path('datasets/acupoint/labels/val')

    img_path = next(img_dir.glob("*.png"))
    label_path = label_dir / (img_path.stem + ".txt")

    print(f"   测试图片: {img_path.name}")
    print(f"   测试标签: {label_path.name}")

    # 读取图片（必须用 IMREAD_UNCHANGED）
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    print(f"   图片形状: {img.shape}")
    assert img.ndim == 3 and img.shape[2] == 4, "❌ 图片不是4通道！"
    print("   ✅ 图片是4通道 RGBD")

    # 读取标签
    with open(label_path) as f:
        line = f.readline().strip()
    parts = list(map(float, line.split()))
    print(f"   标签长度: {len(parts)}")

    # 解析标签
    if len(parts) > 5:
        kpt_part = parts[5:]
        num_kpts = len(kpt_part) // 3
        print(f"   关键点数量: {num_kpts}")
        assert num_kpts == 3, f"❌ 关键点数量不对！期望3，实际{num_kpts}"
    print("   ✅ 标签验证通过")

    # ================== 3. 可视化验证（最直观！） ==================
    print("\n[3/3] 生成可视化验证图...")
    # 拆分 RGB 和深度
    b, g, r, d = cv2.split(img)
    rgb = cv2.merge([r, g, b])  # 转成RGB显示
    depth_colored = cv2.applyColorMap(cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                                      cv2.COLORMAP_JET)

    # 画标签
    vis_img = rgb.copy()
    H, W = vis_img.shape[:2]

    # 画大框
    cls, cx, cy, bw, bh = parts[:5]
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 画关键点
    kpts = np.array(kpt_part).reshape(-1, 3)
    for i, (kx, ky, v) in enumerate(kpts):
        kx_px = int(kx * W)
        ky_px = int(ky * H)
        color = (0, 0, 255) if i == 0 else (255, 0, 0) if i == 1 else (0, 255, 255)
        cv2.circle(vis_img, (kx_px, ky_px), 5, color, -1)
        cv2.putText(vis_img, str(i), (kx_px + 5, ky_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 拼接保存
    top_row = np.hstack([vis_img, depth_colored])
    cv2.imwrite('test_visualization.jpg', top_row)
    print("   ✅ 可视化图已生成: test_visualization.jpg")
    print("   👉 请打开这张图，确认：")
    print("      1. 绿色框是否框住了腹部")
    print("      2. 三个彩色点是否在正确的穴位位置")

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！可以放心试训10轮了！")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_simple()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
```



<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775638755894-1a495770-db28-474b-bc53-e59c7e320939.png)



### 训练脚本
先训练10次，看结果

```python
"""直接使用Python API训练 RGBD-Pose (关键点) 模型"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("🚀 【试训版】RGBD 4通道 Pose(关键点) 模型 🚀")
    print("=" * 60)

    # 1. 加载模型
    print("\n1. 加载模型...")
    model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')

    # 验证模型是 4 通道
    first_layer = model.model.model[0].conv
    print(f"第一层卷积输入通道: {first_layer.weight.shape[1]}")
    assert first_layer.weight.shape[1] == 4, "❌ 模型不是4通道！"
    print("✓ 模型确认为 4 通道")

    # 2. 开始试训 (先跑10轮确认没问题)
    print("\n2. 开始试训...")

    try:
        results = model.train(
            task='pose',
            data='datasets/acupoint.yaml',

            # 【关键】先试训10轮！没问题再改成300
            epochs=10,

            imgsz=640,
            batch=4,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            name='train_pose_rgbd_10ep_test',  # 试训专用名字
            project='runs/pose',

            patience=0,
            save=True,
            plots=True,
            verbose=True,
            workers=0,
            cache=False,
            amp=False,

            # 【核心】100%禁用所有破坏小目标/关键点的增强
            mosaic=0.0,
            copy_paste=0.0,
            mixup=0.0,
            fliplr=0.0,      # 先完全关闭翻转
            flipud=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,

            # 优化器 (小目标用稍小的学习率更稳)
            lr0=0.0005,     # 从0.001降到0.0005
            lrf=0.01,
            warmup_epochs=10, # 热身稍长一点
        )

        print("\n" + "=" * 60)
        print("✅ 试训完成！")
        print("=" * 60)
        print("\n👉 下一步：")
        print("   1. 查看 runs/pose/train_pose_rgbd_10ep_test/results.png")
        print("   2. 确认 box_loss 和 pose_loss 正常下降")
        print("   3. 没问题的话，把 epochs 改成 300 正式训练！")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
```



### 推理
```python
import cv2
import numpy as np
from ultralytics import YOLO

# 加载训练好的4通道模型
model = YOLO('runs/pose/train_pose_rgbd_10ep_test/weights/best.pt', task='pose')

# 读取4通道RGBD测试图
img_path = "H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint/images/val/20250930_103_2_0045.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
assert img.shape[2] == 4, "必须是4通道RGBD图片"

# 推理
results = model(img, conf=0.5)

# 可视化结果
result = results[0]
vis_img = img[..., :3].copy()  # 取RGB通道显示
H, W = vis_img.shape[:2]

# 画腹部大框
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 画穴位关键点
for kpts in result.keypoints:
    for i, (kx, ky, conf) in enumerate(kpts.data[0]):
        kx_px, ky_px = int(kx), int(ky)
        # 不同点用不同颜色
        color = (0, 0, 255) if i == 0 else (255, 0, 0) if i == 1 else (0, 255, 255)
        cv2.circle(vis_img, (kx_px, ky_px), 5, color, -1)
        cv2.putText(vis_img, f"pt{i}", (kx_px+5, ky_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存结果
cv2.imwrite("pose_result.jpg", vis_img)
print("✅ 推理完成，结果已保存为 pose_result.jpg")
```



<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1775639507265-7db33de5-320e-42f6-8dbf-668699ee562f.png)



10次就测通了，有点不可思议，虽然最后两个穴位有点误差



### 加大训练次数重新测试
训练100次，得到的结果还可以，查看训练的数据

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1777454740384-ea44fe97-d21f-4ef7-968c-af65511e7bc0.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1777454825303-90b10748-b418-419d-9bc0-deb6d8571d81.png)





推理测试一下

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1777456780007-c9342d4a-e756-43c5-81a1-194d290a4d4d.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1777456828932-08968e22-ad83-48e6-8877-9a99a6a42ddd.png)



这是拿训练集的数据去做的推理，同理如果拿的是验证集数据去做去推理，像素差在30左右，还是比较大的；

应该是过拟合了，在训练的时候我把早停和增强关闭了，打开早停和数据增强重新测试

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1777457138210-1da6c42b-6135-4c1c-bebe-bc8a1e5a1bd8.png)





## 针对过拟合的重新训练
### 数据集重新拆分
很明显我并没有拆出测试集，而是全部一股脑将数据集拆分成了训练集和验证集都塞给他了；

按照下述的方式进行分开

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1779247557900-6fccc57b-e079-4400-ae5c-e1c703539c0f.png)



最终数据集拆分结果如下图所示

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1779247579661-a652566c-c18b-4aa5-a799-5861bac7c17f.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1779247588438-88e6a433-e236-4184-a605-4b963aa4a7dd.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/54962239/1779247596561-cae0162f-e1b6-41c6-bb20-10eb37b87766.png)





### 添加测试集的配置
```yaml
# 1. 数据集绝对路径
path: H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint
train: images/train
val: images/val
test: images/test   # 🔑 新增这一行！告诉模型你的“期末高考试卷”放在哪里

# 2. 类别配置 (正确，大框是腹部)
nc: 1
names:
  0: abdomen

# 3. RGBD 配置 
rgbd: true
channels: 4

# 4. 🎯 Pose 模型核心配置
kpt_shape: [3, 3]  # 3个关键点，每个点(x,y,visibility)

# 5. 🔑 翻转索引：左右翻转时左↔右互换，肚脐不变
flip_idx: [2, 1, 0]
```



### 重新训练
```python
"""直接使用Python API训练 RGBD-Pose (关键点) 模型"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("🚀 【试训版】RGBD 4通道 Pose(关键点) 模型 🚀")
    print("=" * 60)

    # 1. 加载模型
    print("\n1. 加载模型...")
    model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')

    # 验证模型是 4 通道
    first_layer = model.model.model[0].conv
    print(f"第一层卷积输入通道: {first_layer.weight.shape[1]}")
    assert first_layer.weight.shape[1] == 4, "❌ 模型不是4通道！"
    print("✓ 模型确认为 4 通道")

    # 2. 开始试训 (先跑10轮确认没问题)
    print("\n2. 开始试训...")

    try:
        results = model.train(
            task='pose',
            data='datasets/acupoint.yaml',

            # 【关键】先试训10轮！没问题再改成300
            epochs=10,

            imgsz=640,
            batch=4,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            name='train_pose_rgbd_10ep_test',  # 试训专用名字
            project='runs/pose',

            patience=0,
            save=True,
            plots=True,
            verbose=True,
            workers=0,
            cache=False,
            amp=False,

            # 【开启安全的数据增强】对抗过拟合！
            mosaic=0.0,  # 绝对保持 0！(拼图会破坏腹部的整体拓扑结构)
            mixup=0.0,  # 绝对保持 0！(图像混合会干扰深度图通道)
            copy_paste=0.0,  # 绝对保持 0！

            # --- 下面这些可以安全开启，逼模型泛化 ---
            fliplr=0.5,  # 开启50%概率的左右翻转！(因为你已经写了 flip_idx，现在非常安全)
            flipud=0.0,  # 保持 0 (人不会倒着长)
            degrees=10.0,  # 允许正负 10 度的微小旋转 (模拟人躺得有点歪)
            translate=0.1,  # 允许 10% 的平移 (防止模型死记硬背肚脐在正中心)
            scale=0.2,  # 允许放大缩小 20% (模拟镜头远近变化)
            shear=0.0,  # 保持 0 (避免严重形变)
            perspective=0.0,  # 保持 0 (避免3D透视形变破坏深度计算)

            # 色彩增强保持 0 (防止底层库在处理 4 通道色彩空间转换时把深度图搞坏)
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,

            # 优化器 (小目标用稍小的学习率更稳)
            lr0=0.0005,     # 从0.001降到0.0005
            lrf=0.01,
            warmup_epochs=10, # 热身稍长一点
        )

        print("\n" + "=" * 60)
        print("✅ 试训完成！")
        print("=" * 60)
        print("\n👉 下一步：")
        print("   1. 查看 runs/pose/train_pose_rgbd_10ep_test/results.png")
        print("   2. 确认 box_loss 和 pose_loss 正常下降")
        print("   3. 没问题的话，把 epochs 改成 300 正式训练！")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
```





### 用测试集进行测试
```python
from ultralytics import YOLO

def run_final_test():
    print("============================================================")
    print(" 终极摸底考试：测试集绝密测验 ")
    print("============================================================")

    # 1. 加载你 300 轮训练出来的最终 best.pt
    # ⚠️ 注意：一定要把下面的路径换成你 300 轮跑完后实际生成的路径！
    # 比如可能是 'runs/pose/train_pose_rgbd_300ep/weights/best.pt'
    model = YOLO('runs/pose/train_pose_rgbd_10ep_test3/weights/best.pt', task='pose')

    # 2. 启动验证模式，并强行指定只考 test 集
    print("\n 开始进行测试集评估...")
    metrics = model.val(
        data='datasets/acupoint.yaml',
        split='test',  # 🔑 核心指令：只跑 test 文件夹！
        imgsz=640,
        batch=4,
        device='cuda:0'
    )

    # 3. 打印最终成绩单
    print("\n============================================================")
    print(" 考试成绩单出炉！")
    print(f" 大肚子框准确率 (Box mAP50): {metrics.box.map50:.4f}")
    print(f" 穴位点准确率 (Pose mAP50): {metrics.pose.map50:.4f}")
    print("============================================================")

if __name__ == '__main__':
    run_final_test()
```



**测试结果**

```python
(yolov8) H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection>python YJJ-Pose-Scripts/eval_pipeline_smoke_test.py
============================================================
 终极摸底考试：测试集绝密测验
============================================================

 开始进行测试集评估...
Ultralytics 8.3.176  Python-3.8.20 torch-2.4.1+cu121 CUDA:0 (NVIDIA GeForce RTX 4080 SUPER, 16375MiB)
YOLOv8n-pose summary (fused): 81 layers, 3,077,966 parameters, 0 gradients, 8.4 GFLOPs
val: Fast image access  (ping: 0.20.0 ms, read: 1367.0761.3 MB/s, size: 1212.5 KB)
val: Scanning H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection\datasets\acupoint\labels\test... 14 images, 0 backgrounds, 0 corru
val: New cache created: H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection\datasets\acupoint\labels\test.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP
                   all         14         14      0.996          1      0.995      0.792      0.996          1      0.995      0.995
Speed: 3.3ms preprocess, 18.1ms inference, 0.0ms loss, 9.3ms postprocess per image
Results saved to runs\pose\val

============================================================
 考试成绩单出炉！
 大肚子框准确率 (Box mAP50): 0.9950
 穴位点准确率 (Pose mAP50): 0.9950
============================================================

(yolov8) H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection>
```



## 复盘
### 撞墙期
#### 准备初始工具和数据
操作：

我从网上拉去了一个别人写好的yolo-rgbd-detection项目，然后配置了项目环境，运行脚本生成了一个4通道初始权重；

接着我把分开的RGB彩色图和Depth深度图融合在一起，放进了训练文件夹；



原因：

因为yolo官方，只吃3通道的彩色图，我现在要添加深度信息，但是底层的读取图片的格式等等，单靠我自己是修改不好的；为了确保，图片在走到权重进行识别的这一步格式一定是RGBD的，所以我就从网上用别人修改好的通用的4通道框架；



#### 第一次失败
操作：

在训练时我发现box_loss变成了无穷大，后面发现是穴位的尺寸太小了；于是我写了一个脚本把标签框强行放大到5%, 但测试的时候，模型连瞎蒙都蒙不出来，只画了0个框；



为什么会失败：

因为我一开始用的是目标检测，目标检测的原理是寻找画面里有明显边缘或颜色差异的东西(比如说找一只猫);

但是穴位的像素和别的皮肤特征没有任何区别；让模型在一片毫无特征的肚皮上找一个不存在视觉边界的点，所以模型找不出来；



### 顿悟期
重新制定战术：

操作：

尝试能不能把穴位的识别变成肚脐的识别，然后按照肚脐的位置，反过来去推演每一个穴位的位置，所以我就整个腹部都框起来了；然后把三个穴位的标签正常的导出出来，但是我这里并没有添加每个肚脐的标签；

也就是说只有腹部的框框和三个穴位的标签；



### 执行期
#### 重新下载pose模型
操作：

加载pose模型，强行改成4通道，1个类别，3个关键点；

把模型从目标检测替换到关键点检测；



#### 重新打标签并修改YAML
操作：

只增加了一个腹部的框的标签，然后在yaml中添加flip_idx: [2,1,0]



原因：

flip_idx叫做反转规则，为了让模型见多时光，系统在训练时会把图片左右翻转，如果不加这一行模型学到的左右结构就是乱的了；



### 疑惑
 我不明白，我最后都没有标记肚脐的位置，只添加了框框的坐标，我就在想，如果说我把框框的标签去掉，只保留穴位的标签，训练效果是不是也会这么好呢？



pose模型需要先找大区域，再找小细节。大框的意义是告诉模型我要找的目标在和这个框框中。

当模型成功锁定框框之后，会专门在这个框框里面找目标。

此时我虽然没有标记肚脐，但是模型会自动根据肚脐这个深坑作为参照物，去推导穴位的位置；



如果我换成了别的图片，可能腿的穴位，可能是手部的，我又该怎么做呢？是不是也框住穴位所在的区域，然后导出对应的标签，最后导出新的权重，用素材重新训练呢？



需要重新划定大框，不要只框一个手腕，要框住整个目标所在的结构(比如整个小臂/或整个小腿)

然后寻找新的锚点，深度图对骨骼的凸起和关节的凹陷最敏感；

然后就是修改数据集和以及yaml等，重新下载权重进行训练；





