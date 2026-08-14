"""
LoRA 微调训练脚本

策略:
- 视觉编码器: 冻结（保留 CLIP/SigLIP 通用视觉特征）
- 投影层: 全量微调（适配缺陷域的视觉-语言对齐）
- 语言模型: LoRA 微调（r=8, alpha=16）

运行: python training/train_lora.py --model_path ./models/qwen-vl-chat --train_data ./data/train_data.json --output_dir ./outputs/lora
"""
import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class DiagnosisDataset(Dataset):
    """诊断训练数据集"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"加载训练数据: {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        conversations = item["conversations"]

        user_msg = conversations[0]["content"]
        assistant_msg = conversations[1]["content"]

        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": user_msg},
        ])

        text_input = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        labels = self.tokenizer(
            assistant_msg,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        return {
            "input_ids": text_input["input_ids"].squeeze(0),
            "attention_mask": text_input["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
            "image_path": image_path,
        }


def setup_lora(model, r: int = 8, alpha: int = 16):
    """配置 LoRA"""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "w1", "w2"],
        bias="none",
    )

    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, lora_config)

    for name, param in model.named_modules():
        if "visual" in name.lower() or "vision" in name.lower():
            for p in param.parameters(recurse=True):
                p.requires_grad = False

    model.print_trainable_parameters()
    return model


def train(
    model_path: str,
    train_data: str,
    output_dir: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lr: float = 2e-4,
    epochs: int = 3,
    batch_size: int = 2,
    max_length: int = 1024,
):
    """LoRA 微调训练"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("配置 LoRA...")
    model = setup_lora(model, r=lora_r, alpha=lora_alpha)

    dataset = DiagnosisDataset(train_data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    print(f"\n开始训练: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"训练数据: {len(dataset)} 条")
    print("=" * 60)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"E{epoch+1} B{batch_idx}/{len(dataloader)} | loss={loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{epochs} | avg_loss={avg_loss:.4f}")
        print("-" * 60)

    print("保存 LoRA adapter...")
    model.save_pretrained(str(output_path / "lora_adapter"))
    tokenizer.save_pretrained(str(output_path / "lora_adapter"))
    print(f"已保存到 {output_path / 'lora_adapter'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 微调训练")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen-VL 模型路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据JSON路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    train(
        model_path=args.model_path,
        train_data=args.train_data,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
