"""
模板化诊断描述标注生成

核心设计：基于域知识的模板 + NEU-DET bbox 自动填充变量
- 每类缺陷有一个固定模板（来自 configs/defect_types.json）
- 变量（位置/面积/严重等级）从 bbox 坐标自动计算
- 人工抽检 10% 确认质量

运行: python data/generate_labels.py --data_dir ./data/neu-det --output ./data/train_data.json
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_defect_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_to_position(bbox: List[float], img_w: int, img_h: int) -> str:
    """根据 bbox 坐标计算位置描述"""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2

    h_pos = "左" if cx < img_w / 3 else ("右" if cx > img_w * 2 / 3 else "中")
    v_pos = "上" if cy < img_h / 3 else ("下" if cy > img_h * 2 / 3 else "中")

    if h_pos == "中" and v_pos == "中":
        return "图像中央区域"
    elif h_pos == "中":
        return f"图像{v_pos}部"
    elif v_pos == "中":
        return f"图像{h_pos}侧"
    else:
        return f"图像{v_pos}{h_pos}侧"


def bbox_to_area_ratio(bbox: List[float], img_w: int, img_h: int) -> float:
    """计算 bbox 面积占比"""
    w, h = bbox[2], bbox[3]
    return round((w * h) / (img_w * img_h) * 100, 1)


def area_ratio_to_severity(area_ratio: float, defect_info: dict) -> int:
    """根据面积占比映射严重等级（基于 defect_types.json 的 severity_criteria）"""
    if area_ratio < 5:
        return 1
    elif area_ratio < 15:
        return 2
    elif area_ratio < 30:
        return 3
    elif area_ratio < 50:
        return 4
    else:
        return 5


def generate_diagnosis(
    defect_type: str,
    bbox: List[float],
    img_w: int,
    img_h: int,
    config: dict,
) -> str:
    """生成单条诊断描述"""
    defect_info = config["defect_types"][defect_type]

    position = bbox_to_position(bbox, img_w, img_h)
    area_ratio = bbox_to_area_ratio(bbox, img_w, img_h)
    severity = area_ratio_to_severity(area_ratio, defect_info)

    severity_desc = defect_info["severity_criteria"][str(severity)]

    report = (
        f"【缺陷类型】{defect_info['name_zh']}（{defect_info['name_en']}）\n"
        f"【位置描述】位于{position}，缺陷区域约占检测区域{area_ratio}%\n"
        f"【形态特征】{defect_info['morphology']}\n"
        f"【严重等级】{severity}级（{severity_desc}）\n"
        f"【可能原因】{defect_info['possible_causes']}\n"
        f"【处置建议】{defect_info['general_suggestion']}"
    )

    return report


def generate_training_data(
    data_dir: str,
    config_path: str,
    output_path: str,
):
    """生成完整的训练数据"""
    config = load_defect_config(config_path)
    data_path = Path(data_dir)

    annotation_file = data_path / "annotations" / "instances_train.json"
    if not annotation_file.exists():
        annotation_files = list((data_path / "annotations").glob("*.json"))
        if annotation_files:
            annotation_file = annotation_files[0]
        else:
            print(f"错误: 找不到标注文件 in {data_path / 'annotations'}")
            return

    print(f"加载标注: {annotation_file}")
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    print(f"类别映射: {cat_map}")

    img_info = {img["id"]: img for img in coco["images"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)

    name_fix = {
        "crazing": "crazing", "inclusion": "inclusion", "patches": "patches",
        "pitted_surface": "pitted_surface", "pitted_surface ": "pitted_surface",
        "rolled-in_scale": "rolled-in_scale", "rolled_in_scale": "rolled-in_scale",
        "scratches": "scratches",
    }

    training_data = []
    count = 0
    for img_id, anns in ann_by_img.items():
        img = img_info[img_id]
        img_w, img_h = img["width"], img["height"]
        img_name = img["file_name"]
        img_path = str(data_path / "images" / img_name)

        for ann in anns:
            cat_name = cat_map.get(ann["category_id"], "")
            defect_type = name_fix.get(cat_name, cat_name)

            if defect_type not in config["defect_types"]:
                continue

            bbox = ann["bbox"]
            diagnosis = generate_diagnosis(
                defect_type, bbox, img_w, img_h, config
            )

            training_data.append({
                "id": f"defect_{count:04d}",
                "image": img_path,
                "defect_type": defect_type,
                "bbox": bbox,
                "img_size": [img_w, img_h],
                "conversations": [
                    {"role": "user", "content": "请分析这张钢材表面缺陷图，给出诊断报告。"},
                    {"role": "assistant", "content": diagnosis},
                ],
            })
            count += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"\n生成 {count} 条训练数据 → {output}")

    type_count = {}
    for item in training_data:
        t = item["defect_type"]
        type_count[t] = type_count.get(t, 0) + 1
    print("各类分布:")
    for t, c in sorted(type_count.items()):
        print(f"  {t}: {c}")

    print(f"\n示例诊断报告:")
    print("-" * 40)
    print(training_data[0]["conversations"][1]["content"])
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模板化诊断描述标注")
    parser.add_argument("--data_dir", type=str, default="./data/neu-det")
    parser.add_argument("--config", type=str, default="./configs/defect_types.json")
    parser.add_argument("--output", type=str, default="./data/train_data.json")
    args = parser.parse_args()

    generate_training_data(args.data_dir, args.config, args.output)
