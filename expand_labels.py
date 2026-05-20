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