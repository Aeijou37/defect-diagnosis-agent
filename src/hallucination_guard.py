"""
幻觉控制模块 — 三层防护

层1: 输入层 — 图像质量检查（拒绝低质量图，避免VLM"猜"）
层2: Prompt层 — 强约束 System Prompt（限定输出格式和缺陷类型范围）
层3: 输出层 — 后处理校验（类型合法性/等级范围/格式完整性/原文复述检测）

运行测试: python tests/test_hallucination.py
"""
import re
import json
from pathlib import Path
from typing import Optional
from PIL import Image


VALID_DEFECT_TYPES = {
    "裂纹": "crazing", "Crazing": "crazing",
    "夹杂": "inclusion", "Inclusion": "inclusion",
    "斑块": "patches", "Patches": "patches",
    "麻点": "pitted_surface", "Pitted Surface": "pitted_surface", "PittedSurface": "pitted_surface",
    "轧入氧化皮": "rolled-in_scale", "Rolled-in Scale": "rolled-in_scale", "RolledInScale": "rolled-in_scale",
    "划痕": "scratches", "Scratches": "scratches",
}

VALID_DEFECT_NAMES = set(VALID_DEFECT_TYPES.keys()) | set(VALID_DEFECT_TYPES.values())

REPORT_FIELDS = ["缺陷类型", "位置描述", "形态特征", "严重等级", "可能原因", "处置建议"]


def check_image_quality(image_path: str, min_size: int = 64, min_brightness: float = 10) -> dict:
    """层1: 输入层 — 图像质量检查"""
    try:
        img = Image.open(image_path)
        w, h = img.size
        gray = img.convert("L")
        pixels = list(gray.getdata())
        brightness = sum(pixels) / len(pixels)

        issues = []
        if w < min_size or h < min_size:
            issues.append(f"图像尺寸过小: {w}x{h} (最小要求 {min_size}x{min_size})")
        if brightness < min_brightness:
            issues.append(f"图像亮度过低: {brightness:.1f} (最小要求 {min_brightness})")
        if brightness > 245:
            issues.append(f"图像亮度过高: {brightness:.1f} (可能为空白图)")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "size": [w, h],
            "brightness": round(brightness, 1),
        }
    except Exception as e:
        return {"valid": False, "issues": [f"图像加载失败: {e}"], "size": None, "brightness": None}


def get_system_prompt() -> str:
    """层2: Prompt层 — 强约束 System Prompt"""
    return """你是一个工业缺陷诊断专家。分析钢材表面缺陷图像时必须遵守以下规则：

1. 只能从以下缺陷类型中选择：裂纹(Crazing)、夹杂(Inclusion)、斑块(Patches)、麻点(Pitted Surface)、轧入氧化皮(Rolled-in Scale)、划痕(Scratches)
2. 如果无法确定缺陷类型，必须回答"无法确定，建议人工复检"
3. 严重等级必须是1-5的整数
4. 不得编造图像中不存在的缺陷特征
5. 输出必须严格按以下格式：

【缺陷类型】XXX
【位置描述】XXX
【形态特征】XXX
【严重等级】X级
【可能原因】XXX
【处置建议】XXX"""


def parse_report(report: str) -> dict:
    """解析诊断报告为结构化字典"""
    parsed = {}
    for field in REPORT_FIELDS:
        pattern = f"【{field}】(.*?)(?=【|$)"
        match = re.search(pattern, report, re.DOTALL)
        if match:
            parsed[field] = match.group(1).strip()
        else:
            parsed[field] = None
    return parsed


def validate_defect_type(parsed: dict) -> list:
    """校验缺陷类型是否合法"""
    errors = []
    defect_type = parsed.get("缺陷类型", "")

    if not defect_type:
        errors.append("缺少缺陷类型字段")
        return errors

    if "无法确定" in defect_type:
        return []

    matched = False
    for valid_name in VALID_DEFECT_NAMES:
        if valid_name in defect_type:
            matched = True
            break

    if not matched:
        errors.append(f"无效缺陷类型: {defect_type}（不在预定义的6类中）")

    return errors


def validate_severity(parsed: dict) -> list:
    """校验严重等级是否合法"""
    errors = []
    severity = parsed.get("严重等级", "")

    if not severity:
        errors.append("缺少严重等级字段")
        return errors

    match = re.search(r"(\d)", severity)
    if not match:
        errors.append(f"无法解析等级数字: {severity}")
        return errors

    level = int(match.group(1))
    if not (1 <= level <= 5):
        errors.append(f"等级超出范围: {level}（应为1-5）")

    return errors


def validate_format(parsed: dict) -> list:
    """校验报告格式是否完整"""
    errors = []
    for field in REPORT_FIELDS:
        if parsed.get(field) is None:
            errors.append(f"缺少字段: 【{field}】")
    return errors


def detect_hallucination(parsed: dict) -> list:
    """检测可能的幻觉内容"""
    warnings = []
    morphology = parsed.get("形态特征", "") or ""
    cause = parsed.get("可能原因", "") or ""

    hallucination_keywords = [
        "可能存在", "似乎有", "疑似", "也许", "大概",
        "不确定是否", "可能是", "也许存在",
    ]

    for kw in hallucination_keywords:
        if kw in morphology:
            warnings.append(f"形态特征含不确定描述: '{kw}'")
        if kw in cause and "可能" not in kw:
            warnings.append(f"原因分析含不确定描述: '{kw}'")

    return warnings


def validate_report(report: str) -> dict:
    """层3: 输出层 — 完整后处理校验"""
    parsed = parse_report(report)

    errors = []
    errors.extend(validate_format(parsed))
    errors.extend(validate_defect_type(parsed))
    errors.extend(validate_severity(parsed))

    warnings = detect_hallucination(parsed)

    is_refused = "无法确定" in (parsed.get("缺陷类型") or "")

    return {
        "valid": len(errors) == 0,
        "is_refused": is_refused,
        "errors": errors,
        "warnings": warnings,
        "parsed": parsed,
    }


def post_process(report: str) -> str:
    """后处理清洗：去除指令泄露残留 + 格式补全"""
    leaked_keywords = [
        "你是一个", "请遵守", "System Prompt", "system prompt",
        "你必须", "以下规则", "严格按照",
    ]

    cleaned = report
    for kw in leaked_keywords:
        cleaned = cleaned.replace(kw, "")

    parsed = parse_report(cleaned)
    for field in REPORT_FIELDS:
        if parsed.get(field) is None:
            cleaned += f"\n【{field}】未能提取"

    return cleaned.strip()


def full_pipeline(image_path: str, raw_response: str) -> dict:
    """三层防护完整流水线"""
    quality = check_image_quality(image_path)
    if not quality["valid"]:
        return {
            "accepted": False,
            "reason": "输入层拒绝: " + "; ".join(quality["issues"]),
            "quality": quality,
        }

    cleaned = post_process(raw_response)
    validation = validate_report(cleaned)

    return {
        "accepted": validation["valid"],
        "quality": quality,
        "cleaned_report": cleaned,
        "validation": validation,
    }
