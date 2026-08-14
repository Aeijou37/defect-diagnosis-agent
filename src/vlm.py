"""
VLM 模块 — Qwen-VL 加载与诊断推理

职责：
1. 加载 Qwen-VL-Chat（支持原始权重或 LoRA adapter）
2. 输入缺陷图 → 输出诊断报告
3. 支持多轮对话（用户追问）

回家后运行:
  python -c "from src.vlm import VLMWrapper; v = VLMWrapper('./models/qwen-vl-chat'); print(v.diagnose('test.jpg'))"
"""
import torch
from pathlib import Path
from typing import Optional
from PIL import Image


SYSTEM_PROMPT = """你是一个工业缺陷诊断专家。分析钢材表面缺陷图像时必须遵守以下规则：

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


class VLMWrapper:
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"加载 VLM: {self.model_path}")
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config,
            ).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).eval()

        if self.lora_path:
            from peft import PeftModel
            print(f"加载 LoRA adapter: {self.lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)

        print(f"模型加载完成 (4bit={self.load_in_4bit}, LoRA={self.lora_path is not None})")

    def diagnose(self, image_path: str, question: str = None) -> str:
        """输入缺陷图，输出诊断报告"""
        if question is None:
            question = "请分析这张钢材表面缺陷图，给出诊断报告。"

        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": question},
        ])

        response, _ = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
        )
        return response

    def diagnose_with_system_prompt(self, image_path: str, question: str = None) -> str:
        """带强约束 System Prompt 的诊断推理"""
        if question is None:
            question = "请分析这张钢材表面缺陷图，给出诊断报告。"

        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": f"{SYSTEM_PROMPT}\n\n{question}"},
        ])

        response, _ = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
        )
        return response

    def chat(self, image_path: str, question: str, history: list) -> str:
        """多轮对话（用户追问）"""
        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": question},
        ])
        response, new_history = self.model.chat(
            self.tokenizer,
            query=query,
            history=history,
        )
        return response, new_history

    def diagnose_batch(self, image_paths: list, questions: list = None) -> list:
        """批量诊断"""
        results = []
        for i, img_path in enumerate(image_paths):
            q = questions[i] if questions else None
            result = self.diagnose_with_system_prompt(img_path, q)
            results.append(result)
            print(f"[{i+1}/{len(image_paths)}] {Path(img_path).name}: {result[:50]}...")
        return results
