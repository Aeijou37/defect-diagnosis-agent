"""
诊断 Agent 主逻辑

流程: 图像预处理 → VLM推理 → 幻觉控制 → 输出诊断报告
"""
import json
from pathlib import Path
from typing import Optional
from src.vlm import VLMWrapper
from src.hallucination_guard import (
    check_image_quality,
    get_system_prompt,
    validate_report,
    post_process,
    full_pipeline,
)


class DiagnosisAgent:
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = False,
        config_path: str = "configs/defect_types.json",
    ):
        self.vlm = VLMWrapper(
            model_path=model_path,
            lora_path=lora_path,
            device=device,
            load_in_4bit=load_in_4bit,
        )
        self.config_path = config_path
        self.system_prompt = get_system_prompt()

    def diagnose(self, image_path: str, question: str = None) -> dict:
        """完整诊断流程"""
        if question is None:
            question = "请分析这张钢材表面缺陷图，给出诊断报告。"

        result = {
            "image": image_path,
            "accepted": False,
            "quality_check": None,
            "raw_response": "",
            "cleaned_report": "",
            "validation": None,
            "error": None,
        }

        quality = check_image_quality(image_path)
        result["quality_check"] = quality
        if not quality["valid"]:
            result["error"] = "输入层拒绝: " + "; ".join(quality["issues"])
            return result

        raw_response = self.vlm.diagnose_with_system_prompt(image_path, question)
        result["raw_response"] = raw_response

        pipeline_result = full_pipeline(image_path, raw_response)

        result["accepted"] = pipeline_result["accepted"]
        result["cleaned_report"] = pipeline_result.get("cleaned_report", "")
        result["validation"] = pipeline_result.get("validation", None)

        return result

    def diagnose_batch(self, image_paths: list) -> list:
        """批量诊断"""
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] {Path(img_path).name}")
            result = self.diagnose(img_path)
            results.append(result)

            if result["accepted"]:
                print(f"  ✅ 诊断完成")
                print(f"  {result['cleaned_report'][:100]}...")
            else:
                print(f"  ❌ 拒绝: {result.get('error', '校验失败')}")

        return results

    def chat(self, image_path: str, question: str, history: list = None) -> dict:
        """多轮对话（用户追问）"""
        if history is None:
            history = []

        response, new_history = self.vlm.chat(image_path, question, history)
        cleaned = post_process(response)
        validation = validate_report(cleaned)

        return {
            "response": cleaned,
            "valid": validation["valid"],
            "validation": validation,
            "history": new_history,
        }

    def save_results(self, results: list, output_path: str):
        """保存诊断结果"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output}")
