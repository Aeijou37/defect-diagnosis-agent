# 工业缺陷多模态诊断 Agent

> 🔨 开发中 | 将传统缺陷分类（输出类别标签）升级为自然语言诊断报告生成
> 把 CXMT 的"缺陷图→分类数字"升级为"缺陷图→诊断报告（类型+位置+严重度+原因+建议）"

---

## 项目动机

在 CXMT 实习时，我做的是缺陷分类——输入缺陷图，输出一个类别数字。但产线工程师需要的不是数字，而是一份诊断报告：这是什么缺陷、多严重、为什么会产生、应该怎么处理。

这个项目用 VLM（视觉语言模型）把分类升级为**自动诊断报告生成**。

> 详细技术方案见：[项目方案文档](项目方案.md)

## 核心技术点

1. **长尾数据构造** — NEU-DET 钢材缺陷数据集（1800张，6类真实标签）+ 基于域知识的模板化诊断描述标注
2. **VLM LoRA 微调** — Qwen-VL-Chat + LoRA（r=8），视觉编码器冻结，语言模型学习工业缺陷专业表达
3. **幻觉控制** — 三层防护（输入质量检查 + 强约束 System Prompt + 输出后处理校验）
4. **三维评估** — 分类准确率（vs NEU-DET GT）+ 位置准确率（vs bbox）+ 幻觉率（人工检查）
5. **对比实验** — 同一数据集跑纯分类（ResNet50）vs VLM 诊断，证明 VLM 诊断的业务价值

## 数据方案

| 来源 | 数量 | 说明 |
|---|---|---|
| NEU-DET（公开） | 1800张 | 钢材表面缺陷，6类，每类300张，有真实标签+bbox |
| 诊断描述标注 | 1800对 | 基于域知识的模板化标注（我自己写，不用GPT-4o猜） |

**6类缺陷**：crazing（裂纹）/ inclusion（夹杂）/ patches（斑块）/ pitted_surface（麻点）/ rolled-in_scale（轧入氧化皮）/ scratches（划痕）

## 技术栈

| 组件 | 选型 |
|---|---|
| 基础 VLM | Qwen-VL-Chat（9B，4bit 量化） |
| 微调 | PEFT LoRA（r=8, alpha=16） |
| 数据集 | NEU-DET |
| 评估 | 分类准确率 + GPT-4o 评分 + 人工幻觉检查 |
| 前端 | Gradio |
| 部署 | HuggingFace Space |

## 开发计划

| 周 | 目标 | 状态 |
|---|---|---|
| 第1周 | 数据构造：NEU-DT 下载 + 模板化标注 | 🔲 未开始 |
| 第2周 | VLM LoRA 微调 + 初步评估 | 🔲 未开始 |
| 第3周 | 幻觉控制 + 三维评估框架 | 🔲 未开始 |
| 第4周 | Gradio demo + 部署 + README + 技术博客 | 🔲 未开始 |

## 和已有经验的关系

| 经历 | 在本项目中的迁移 |
|---|---|
| CXMT 缺陷分类 | 缺陷域知识 + 长尾处理经验 + 对比实验设计 |
| 讯飞 X射线评片 | 缺陷分析方法（类型+位置+严重度+原因） |
| RAG 学术问答 | 检索增强（可选：缺陷知识库辅助诊断） |

## 硬件环境

| 硬件 | 用途 |
|---|---|
| 4060 Ti 16GB | 日常开发 + 4bit 量化推理 |
| 4卡 P40 24GB | LoRA 微调（大显存 fp16） |
| 4090 24GB | 短期加速（可选） |

## 面试故事线

> "我在 CXMT 做缺陷分类，97.79%→100%，但输出只是一个数字。产线工程师需要的是诊断报告。所以我自己做了这个项目，用 VLM 把分类升级为诊断报告生成。数据用公开的 NEU-DET 钢材缺陷数据集，我自己写了专业诊断描述——这些描述基于我在工业 AI 实习中学到的缺陷分析方法，不是 GPT-4o 猜的。同时做了对比实验：同样的数据跑纯分类和 VLM 诊断，证明 VLM 诊断在保持分类准确率的同时，输出信息量和业务价值显著提升。"

---

## 项目结构

```
defect-diagnosis-agent/
├── README.md
├── requirements.txt
├── 项目方案.md                      # 详细技术方案
├── configs/
│   └── defect_types.json            # 6类缺陷定义（形态/原因/等级标准/建议）
├── data/
│   ├── prepare_neudet.py            # NEU-DET 数据集下载与整理
│   └── generate_labels.py           # 模板化诊断描述标注生成
├── src/
│   ├── vlm.py                       # VLM 加载与诊断推理
│   ├── hallucination_guard.py       # 幻觉控制（三层防护）
│   ├── diagnosis_agent.py           # 诊断 Agent 主逻辑
│   └── app.py                       # Gradio 前端
├── training/
│   └── train_lora.py                # LoRA 微调训练
├── evaluation/
│   └── eval_all.py                  # 三维评估（准确率+位置+幻觉率）
└── tests/
    └── test_hallucination.py        # 幻觉控制单元测试
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 NEU-DET 数据集

```bash
python data/prepare_neudet.py --data_dir ./data/neu-det
# 按提示手动下载数据集后:
python data/prepare_neudet.py --data_dir ./data/neu-det --organize
```

### 3. 生成诊断标注

```bash
python data/generate_labels.py --data_dir ./data/neu-det --output ./data/train_data.json
```

### 4. 下载 Qwen-VL 模型

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen-VL-Chat-Int4', local_dir='./models/qwen-vl-chat')
"
```

### 5. 测试幻觉控制（不需要模型）

```bash
python tests/test_hallucination.py
```

### 6. LoRA 微调（需要 GPU）

```bash
python training/train_lora.py \
    --model_path ./models/qwen-vl-chat \
    --train_data ./data/train_data.json \
    --output_dir ./outputs/lora \
    --epochs 3 --batch_size 2
```

### 7. 启动诊断应用

```bash
# 零样本（不微调）
python src/app.py ./models/qwen-vl-chat

# LoRA 微调后
python src/app.py ./models/qwen-vl-chat ./outputs/lora/lora_adapter
```

浏览器打开 `http://localhost:7860`

### 8. 评估

```bash
python evaluation/eval_all.py \
    --results ./results/diagnosis_results.json \
    --gt ./data/neu-det/annotations/instances_test.json \
    --output ./results/eval_report.json
```

---

> 📌 项目完成后将撰写技术笔记，发布到 [cv-algorithm-notes](https://github.com/Aeijou37/cv-algorithm-notes) 作为第05篇。
