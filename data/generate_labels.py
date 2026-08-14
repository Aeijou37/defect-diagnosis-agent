"""
NEU-DET 数据集下载与预处理

NEU-DET: 1800张钢材表面缺陷图像，6类，每类300张
来源: 东北大学(NEU) 钢材表面缺陷数据集
下载: https://university.roboflow.com/ 或者通过 kaggle/huggingface

回家后运行: python data/prepare_neudet.py --data_dir ./data/neu-det
"""
import os
import sys
import argparse
import shutil
from pathlib import Path


def download_neudet(data_dir: str):
    """下载 NEU-DET 数据集"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    images_dir = data_path / "images"
    annotations_dir = data_path / "annotations"

    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"数据已存在: {images_dir}")
        return

    print("=" * 60)
    print("NEU-DET 数据集下载说明")
    print("=" * 60)
    print()
    print("NEU-DET 数据集需要手动下载，请选择以下任一方式：")
    print()
    print("方式1: Roboflow (推荐)")
    print("  1. 访问 https://university.roboflow.com/ 搜索 NEU-DET")
    print("  2. 下载 COCO 格式")
    print("  3. 解压到 ./data/neu-det/")
    print()
    print("方式2: Kaggle")
    print("  1. 访问 https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
    print("  2. 下载并解压到 ./data/neu-det/")
    print()
    print("方式3: HuggingFace")
    print("  python -c \"")
    print("    from huggingface_hub import snapshot_download")
    print("    snapshot_download(repo_id='Abdoullah/NEU-DET', repo_type='dataset', local_dir='./data/neu-det')")
    print("  \"")
    print()
    print("下载后目录结构应为:")
    print("  ./data/neu-det/")
    print("    ├── images/          # 1800张jpg")
    print("    └── annotations/     # COCO格式JSON (含bbox)")
    print()

    sys.exit(0)


def organize_neudet(data_dir: str):
    """整理数据集为统一格式"""
    data_path = Path(data_dir)
    images_dir = data_path / "images"

    if not images_dir.exists():
        print(f"错误: 找不到 {images_dir}，请先下载数据集")
        return

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"找到 {len(image_files)} 张图像")

    defect_classes = {
        "crazing": "Cr",
        "inclusion": "In",
        "patches": "Pa",
        "pitted_surface": "PS",
        "rolled-in_scale": "RS",
        "scratches": "Sc",
    }

    organized = data_path / "organized"
    for cls_name in defect_classes:
        (organized / cls_name).mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in image_files:
        name = img_path.stem.lower()
        for cls_name, prefix in defect_classes.items():
            if name.startswith(prefix.lower()) or prefix.lower() in name:
                shutil.copy2(img_path, organized / cls_name / img_path.name)
                count += 1
                break

    print(f"已整理 {count} 张图像到 {organized}")
    for cls_name in defect_classes:
        cls_count = len(list((organized / cls_name).glob("*")))
        print(f"  {cls_name}: {cls_count}张")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEU-DET 数据集预处理")
    parser.add_argument("--data_dir", type=str, default="./data/neu-det")
    parser.add_argument("--organize", action="store_true", help="整理数据集")
    args = parser.parse_args()

    if args.organize:
        organize_neudet(args.data_dir)
    else:
        download_neudet(args.data_dir)
