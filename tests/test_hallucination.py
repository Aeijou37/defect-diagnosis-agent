"""
测试脚本 — 幻觉控制模块

不需要 VLM 模型，单独测试三层防护逻辑。
运行: python tests/test_hallucination.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hallucination_guard import (
    check_image_quality,
    parse_report,
    validate_defect_type,
    validate_severity,
    validate_format,
    validate_report,
    post_process,
    detect_hallucination,
)


def test_parse_report():
    """测试报告解析"""
    report = """【缺陷类型】裂纹（Crazing）
【位置描述】位于图像中央区域
【形态特征】表面呈现不规则网状细纹
【严重等级】2级
【可能原因】轧制过程中温度应力
【处置建议】轻微裂纹可打磨处理"""

    parsed = parse_report(report)
    assert parsed["缺陷类型"] == "裂纹（Crazing）"
    assert parsed["严重等级"] == "2级"
    assert "网状细纹" in parsed["形态特征"]
    print("✅ 报告解析测试通过")


def test_validate_type():
    """测试缺陷类型校验"""
    assert len(validate_defect_type({"缺陷类型": "裂纹（Crazing）"})) == 0
    assert len(validate_defect_type({"缺陷类型": "夹杂（Inclusion）"})) == 0
    assert len(validate_defect_type({"缺陷类型": "未知缺陷"})) > 0
    assert len(validate_defect_type({"缺陷类型": "无法确定，建议人工复检"})) == 0
    assert len(validate_defect_type({"缺陷类型": None})) > 0
    print("✅ 缺陷类型校验测试通过")


def test_validate_severity():
    """测试严重等级校验"""
    assert len(validate_severity({"严重等级": "3级"})) == 0
    assert len(validate_severity({"严重等级": "1级"})) == 0
    assert len(validate_severity({"严重等级": "5级"})) == 0
    assert len(validate_severity({"严重等级": "6级"})) > 0
    assert len(validate_severity({"严重等级": "0级"})) > 0
    assert len(validate_severity({"严重等级": None})) > 0
    print("✅ 严重等级校验测试通过")


def test_validate_format():
    """测试格式完整性校验"""
    complete = {"缺陷类型": "裂纹", "位置描述": "中央", "形态特征": "网状",
                "严重等级": "2级", "可能原因": "温度", "处置建议": "打磨"}
    assert len(validate_format(complete)) == 0

    incomplete = {"缺陷类型": "裂纹", "位置描述": None}
    errors = validate_format(incomplete)
    assert len(errors) == 4
    print("✅ 格式完整性校验测试通过")


def test_detect_hallucination():
    """测试幻觉检测"""
    clean = {"形态特征": "表面网状细纹", "可能原因": "温度应力"}
    assert len(detect_hallucination(clean)) == 0

    suspicious = {"形态特征": "可能存在裂纹", "可能原因": "不确定"}
    assert len(detect_hallucination(suspicious)) > 0
    print("✅ 幻觉检测测试通过")


def test_post_process():
    """测试后处理清洗"""
    leaked = "你是一个工业缺陷诊断专家。【缺陷类型】裂纹"
    cleaned = post_process(leaked)
    assert "你是一个" not in cleaned
    assert "【缺陷类型】" in cleaned
    print("✅ 后处理清洗测试通过")


def test_validate_report():
    """测试完整报告校验"""
    good_report = """【缺陷类型】裂纹（Crazing）
【位置描述】位于图像中央区域
【形态特征】表面呈现不规则网状细纹
【严重等级】2级
【可能原因】轧制过程中温度应力
【处置建议】轻微裂纹可打磨处理"""

    result = validate_report(good_report)
    assert result["valid"] is True
    assert result["is_refused"] is False

    bad_report = """【缺陷类型】外星生物入侵
【位置描述】无处不在
【形态特征】疑似有某种东西
【严重等级】99级
【可能原因】也许
【处置建议】烧了"""

    result = validate_report(bad_report)
    assert result["valid"] is False
    assert len(result["errors"]) > 0

    refused_report = "【缺陷类型】无法确定，建议人工复检"
    result = validate_report(refused_report)
    assert result["is_refused"] is True
    print("✅ 完整报告校验测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("幻觉控制模块测试")
    print("=" * 60)
    test_parse_report()
    test_validate_type()
    test_validate_severity()
    test_validate_format()
    test_detect_hallucination()
    test_post_process()
    test_validate_report()
    print("=" * 60)
    print("✅ 全部测试通过")
    print("=" * 60)
