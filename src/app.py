"""
Gradio 前端 — 工业缺陷诊断 Agent 界面

功能:
1. 上传缺陷图 → 显示诊断报告
2. 多轮追问对话
3. 显示三层校验结果

运行: python src/app.py --model_path ./models/qwen-vl-chat --lora_path ./outputs/lora/lora_adapter
"""
import gradio as gr
from pathlib import Path
from src.diagnosis_agent import DiagnosisAgent


class DiagnosisApp:
    def __init__(self, model_path: str, lora_path: str = None, load_in_4bit: bool = True):
        print("初始化诊断 Agent...")
        self.agent = DiagnosisAgent(
            model_path=model_path,
            lora_path=lora_path,
            load_in_4bit=load_in_4bit,
        )
        self.current_image = None
        self.chat_history = []
        print("初始化完成")

    def diagnose(self, image):
        if image is None:
            return "请上传缺陷图像", "", "", ""

        image_path = image if isinstance(image, str) else image.name
        self.current_image = image_path
        self.chat_history = []

        result = self.agent.diagnose(image_path)

        if not result["accepted"]:
            error = result.get("error", "校验失败")
            validation = result.get("validation", {})
            if validation:
                errors = validation.get("errors", [])
                error += "\n校验错误: " + "; ".join(errors)
            return f"❌ 拒绝诊断: {error}", "", "", ""

        report = result["cleaned_report"]
        validation = result["validation"]

        status = f"✅ 诊断完成\n"
        status += f"类型校验: {'通过' if not validation.get('errors') else '失败'}\n"
        status += f"幻觉警告: {len(validation.get('warnings', []))} 条"

        quality = result["quality_check"]
        quality_info = f"尺寸: {quality['size']} | 亮度: {quality['brightness']}"

        return report, status, quality_info, ""

    def chat(self, question, history):
        if not self.current_image:
            return "", history + [(question, "请先上传缺陷图像进行诊断")]

        result = self.agent.chat(self.current_image, question, self.chat_history)
        self.chat_history = result["history"]

        response = result["response"]
        if not result["valid"]:
            response += "\n\n⚠️ 校验提示: " + "; ".join(result["validation"].get("errors", []))

        history.append((question, response))
        return "", history

    def build(self):
        with gr.Blocks(title="工业缺陷多模态诊断 Agent") as demo:
            gr.Markdown("# 工业缺陷多模态诊断 Agent")
            gr.Markdown("上传钢材表面缺陷图，自动生成诊断报告（类型+位置+形态+等级+原因+建议）")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="上传缺陷图", type="filepath")
                    diagnose_btn = gr.Button("开始诊断", variant="primary")
                    quality_output = gr.Textbox(label="图像质量", interactive=False)
                    status_output = gr.Textbox(label="诊断状态", interactive=False)

                with gr.Column(scale=1):
                    report_output = gr.Textbox(label="诊断报告", lines=12, interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 追问对话")
            chatbot = gr.Chatbot(label="对话")
            question_input = gr.Textbox(label="追问", placeholder="如：这个缺陷的严重程度如何判定？")
            chat_btn = gr.Button("发送")

            diagnose_btn.click(
                fn=self.diagnose,
                inputs=[image_input],
                outputs=[report_output, status_output, quality_output, question_input],
            )

            chat_btn.click(
                fn=self.chat,
                inputs=[question_input, chatbot],
                outputs=[question_input, chatbot],
            )

        return demo


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/qwen-vl-chat"
    lora_path = sys.argv[2] if len(sys.argv) > 2 else None
    load_4bit = "--fp16" not in sys.argv

    app = DiagnosisApp(model_path, lora_path, load_in_4bit=load_4bit)
    demo = app.build()
    demo.launch(server_name="0.0.0.0", server_port=7860)
