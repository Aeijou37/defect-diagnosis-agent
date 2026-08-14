"""
三维评估框架

维度1: 分类准确率 — VLM 输出的缺陷类型 vs NEU-DET GT
维度2: 位置准确率 — VLM 输出的位置描述 vs bbox 坐标实际位置
维度3: 幻觉率 — 描述中有多少是图像中实际不存在的（人工检查）

运行: python evaluation/eval_all.py --results ./results/diagnosis_results.json --gt ./data/neu-det/annotations/instances_test.json
"""
import json
import re
from pathlib import Path
from typing import List, Dict


def load_results(results_path: str) -> list:
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gt(gt_path: str) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_info = {img["id"]: img for img in coco["images"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)
    return {"cat_map": cat_map, "img_info": img_info, "ann_by_img": ann_by_img, "coco": coco}


def extract_defect_type(report: str) -> str:
    """从诊断报告中提取缺陷类型"""
    match = re.search(r"【缺陷类型】(.*?)(?=【|$)", report, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_severity(report: str) -> int:
    """从诊断报告中提取严重等级"""
    match = re.search(r"【严重等级】(\d)", report)
    if match:
        return int(match.group(1))
    return -1


def extract_position(report: str) -> str:
    """从诊断报告中提取位置描述"""
    match = re.search(r"【位置描述】(.*?)(?=【|$)", report, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def normalize_type(type_str: str) -> str:
    """归一化缺陷类型名称"""
    type_map = {
        "裂纹": "crazing", "crazing": "crazing", "Crazing": "crazing",
        "夹杂": "inclusion", "inclusion": "inclusion", "Inclusion": "inclusion",
        "斑块": "patches", "patches": "patches", "Patches": "patches",
        "麻点": "pitted_surface", "pitted_surface": "pitted_surface", "Pitted": "pitted_surface",
        "轧入氧化皮": "rolled-in_scale", "rolled-in_scale": "rolled-in_scale", "Rolled": "rolled-in_scale",
        "划痕": "scratches", "scratches": "scratches", "Scratches": "scratches",
    }
    for k, v in type_map.items():
        if k in type_str:
            return v
    return type_str


def bbox_to_position_gt(bbox: list, img_w: int, img_h: int) -> str:
    """从 GT bbox 计算位置（用于位置准确率评估）"""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    h_pos = "左" if cx < img_w / 3 else ("右" if cx > img_w * 2 / 3 else "中")
    v_pos = "上" if cy < img_h / 3 else ("下" if cy > img_h * 2 / 3 else "中")
    if h_pos == "中" and v_pos == "中":
        return "中"
    elif h_pos == "中":
        return v_pos
    elif v_pos == "中":
        return h_pos
    else:
        return f"{v_pos}{h_pos}"


def eval_classification_accuracy(results: list, gt_data: dict) -> dict:
    """维度1: 分类准确率"""
    correct = 0
    total = 0
    per_class = {}

    for result in results:
        if not result.get("accepted"):
            continue

        report = result.get("cleaned_report", "")
        predicted_type = normalize_type(extract_defect_type(report))

        img_name = Path(result["image"]).name
        img_id = None
        for iid, info in gt_data["img_info"].items():
            if info["file_name"] == img_name:
                img_id = iid
                break

        if img_id is None:
            continue

        anns = gt_data["ann_by_img"].get(img_id, [])
        if not anns:
            continue

        gt_type = normalize_type(gt_data["cat_map"][anns[0]["category_id"]])

        if predicted_type not in per_class:
            per_class[predicted_type] = {"correct": 0, "total": 0}
        per_class[predicted_type]["total"] += 1

        if predicted_type == gt_type:
            correct += 1
            per_class[predicted_type]["correct"] += 1

        total += 1

    accuracy = correct / total if total > 0 else 0

    per_class_acc = {}
    for cls, counts in per_class.items():
        per_class_acc[cls] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_acc,
    }


def eval_position_accuracy(results: list, gt_data: dict) -> dict:
    """维度2: 位置准确率"""
    correct = 0
    total = 0

    for result in results:
        if not result.get("accepted"):
            continue

        report = result.get("cleaned_report", "")
        predicted_pos = extract_position(report)

        img_name = Path(result["image"]).name
        img_id = None
        for iid, info in gt_data["img_info"].items():
            if info["file_name"] == img_name:
                img_id = iid
                break

        if img_id is None:
            continue

        anns = gt_data["ann_by_img"].get(img_id, [])
        if not anns:
            continue

        img_info = gt_data["img_info"][img_id]
        gt_pos = bbox_to_position_gt(anns[0]["bbox"], img_info["width"], img_info["height"])

        if gt_pos in predicted_pos or any(k in predicted_pos for k in gt_pos):
            correct += 1
        total += 1

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


def eval_hallucination_rate(results: list) -> dict:
    """维度3: 幻觉率（基于校验结果统计）"""
    total = len(results)
    hallucinated = 0
    refused = 0
        validation = result.get("validation", {})
        if not result.get("accepted"):
            hallucinated += 1
            continue

        if validation.get("is_refused"):
            refused += 1
            continue

        warnings = validation.get("warnings", [])
        if len(warnings) > 0:
            hallucinated += 1

    valid_total = total - refused
    return {
        "hallucination_rate": round(hallucinated / valid_total, 4) if valid_total > 0 else 0,
        "hallucinated": hallucinated,
        "refused": refused,
        "valid_total": valid_total,
        "total": total,
    }


def generate_report(results: list, gt_data: dict) -> dict:
    """生成完整三维评估报告"""
    cls_acc = eval_classification_accuracy(results, gt_data)
    pos_acc = eval_position_accuracy(results, gt_data)
    hall_rate = eval_hallucination_rate(results)

    report = {
        "summary": {
            "total_samples": len(results),
            "classification_accuracy": cls_acc["accuracy"],
            "position_accuracy": pos_acc["accuracy"],
            "hallucination_rate": hall_rate["hallucination_rate"],
        },
        "classification": cls_acc,
        "position": pos_acc,
        "hallucination": hall_rate,
    }

    print("=" * 60)
    print("三维评估报告")
    print("=" * 60)
    print(f"\n总样本数: {len(results)}")
    print(f"\n维度1 - 分类准确率: {cls_acc['accuracy']:.2%} ({cls_acc['correct']}/{cls_acc['total']})")
    for cls, acc in cls_acc["per_class_accuracy"].items():
        print(f"  {cls}: {acc:.2%}")
    print(f"\n维度2 - 位置准确率: {pos_acc['accuracy']:.2%} ({pos_acc['correct']}/{pos_acc['total']})")
    print(f"\n维度3 - 幻觉率: {hall_rate['hallucination_rate']:.2%} ({hall_rate['hallucinated']}/{hall_rate['valid_total']})")
    print(f"  拒绝判断(安全): {hall_rate['refused']}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="三维评估")
    parser.add_argument("--results", type=str, required=True, help="诊断结果JSON")
    parser.add_argument("--gt", type=str, required=True, help="GT标注JSON (COCO格式)")
    parser.add_argument("--output", type=str, default="./results/eval_report.json")
    args = parser.parse_args()

    results = load_results(args.results)
    gt_data = load_gt(args.gt)
    report = generate_report(results, gt_data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存: {args.output}")
