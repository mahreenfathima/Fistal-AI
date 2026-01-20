#بسم الله الرحمن الرحيم
import gradio as gr
import asyncio
import base64
from client import run_fistal
import asyncio
import os
# from dotenv import load_dotenv
# load_dotenv()

REQUIRED_SECRETS = [
    "GOOGLE_API_KEY_1",
    "GOOGLE_API_KEY_2", 
    "GOOGLE_API_KEY_3",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET"
]

missing = [s for s in REQUIRED_SECRETS if not os.getenv(s)]
if missing:
    raise ValueError(f"Missing secrets in HF Space: {', '.join(missing)}\nAdd them in Settings → Variables and secrets")


def image_to_base64(filepath):
    try:
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = "image/jpeg" if filepath.lower().endswith((".jpg", ".jpeg")) else "image/png"
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {filepath}")
        return ""


image_data_url = image_to_base64("static/new.jpg") 
full_img = image_to_base64("static/fullnew.jpg")

css=f"""
.gradio-container {{
        background: url('{full_img}') !important;
        background-size: cover !important;
    }}
    
    .gradio-container .block {{
        background-color: none !important;
    }}

    .gradio-container .wrap {{
    background-color: none !important;
        border: !important;
        box-shadow: none !important;
        outline: none !important;
    }}

    .features-box {{
    padding: 10px;
    color: white !important;
    background-color: #white !important;
    }}
    .features-box .block {{
    border: blue 1px !important;
    background-color: yellow !important;
    }}
    #tuner {{
        background: linear-gradient(to right, #008DDA, #6A1AAB, #C71585, #F56C40) !important;
        color: white !important;
        margin-top: 5px;
    }}
    #flow {{
    padding: 8px !important;
    color: white !important;
    }}
    #flow .markdown-text {{
    color: white !important;
    }}
    .drop li {{
        background-color: #bcb9cf !important;
        color: black !important;
    }}
    .drop input {{
        background-color: #bcb9cf !important;
        background-size: cover !important;
        color: black !important;
        border: none !important;
        padding: 6px 10px !important;
        border-radius: 4px !important;
    }}
    .out {{
    padding: 10px !important;
    font-size: 16px !important;
    color: white !important;
    background: linear-gradient(90deg, rgba(102,126,234,0.3), rgba(106,26,180,0.3), rgba(245,108,64,0.3)) !important;
    border-radius: 10px !important;
}}

.log-container {{
    max-height: 600px !important;
    overflow-y: auto !important;
    background: rgba(14, 15, 15, 0.5) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    border: 1px solid #3a3a3a !important;
}}
   #stat {{
    min-height: 60px !important;
}}

#stat input, #stat textarea {{
    padding: 12px 10px !important;
    line-height: 1.5 !important;
    color: black !important;
    min-height: 60px !important;
    height: auto !important;
    display: flex !important;
    align-items: center !important;
}}

#stat .wrap {{
    min-height: 60px !important;
}}
    .mod {{
    background: linear-gradient(to right, #008DDA, #6A1AAB, #C71585, #F56C40) !important;
    color: white !important;
    }}
    .log-container::-webkit-scrollbar {{
        width: 10px;
    }}
    
    .log-container::-webkit-scrollbar-track {{
        background: #2a2a2a;
        border-radius: 5px;
    }}
    
    .log-container::-webkit-scrollbar-thumb {{
        background: linear-gradient(to bottom, #008DDA, #6A1AAB, #C71585, #F56C40);
        border-radius: 5px;
    }}
    
    .log-container::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(to bottom, #0099ee, #7722bb, #dd1595, #ff7750);
    }}
    #copy-btn {{
    background: linear-gradient(to right, #008DDA, #6A1AAB, #C71585, #F56C40) !important;
    color: white !important;
    margin-top: 10px !important;
}}
    
    
    :root, .gradio-container * {{
    --block-background-fill: #0e0f0f !important;
    --panel-background-fill: none !important;
    --input-background-fill: #bcb9cf !important;
    --color-background-primary: #0e0f0f !important;
    --block-border-width: 0px !important;
    --block-border-color: #27272a !important;
    --panel-border-width: 0px !important;
    --input-text-color: #000000 !important;
    --input-placeholder-color: #27272a !important;
    --panel-border-color: linear-gradient(to right, #008DDA, #6A1AAB, #C71585, #F56C40) !important;
    --neutral-50: #27272a !important;
}}


"""
        


def app():
    with gr.Blocks(title="Fistal AI 🚀") as demo:
        
        # Header section with background image
        gr.HTML(f"""
            <div style="
                background: url('{image_data_url}');
                background-size: cover;
                background-position: center; 
                background-repeat: no-repeat;
                padding: 20px;
                margin-top: 10px;
                margin-bottom: 10px;
                border-radius: 10px;">
                <h1 style="color: white !important; font-size: 35px;">Fistal AI 🚀</h1>
                <p style="color: white; margin-top: -5px;">Seamlessly fine-tune LLMs with an Agentic AI powered by MCP, Modal, and Unsloth.</p>
                <div style="display:flex; gap:5px; flex-wrap:wrap; align-items:center; margin-bottom:15px;">
                    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20-%20HF%20Space%20-%20orange" alt="HF Space">
                    <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
                    <img src="https://img.shields.io/badge/%E2%9C%A8%20-%20Gemini%20API%20-%20teal" alt="Gemini">
                    <img src="https://img.shields.io/badge/Modal-Enabled-green" alt="Modal">
                    <img src="https://img.shields.io/badge/Unsloth-4bit-purple" alt="Unsloth">
                    <img src="https://img.shields.io/badge/MCP-Enabled-pink" alt="MCP">
                    <img src="https://img.shields.io/badge/%F0%9F%94%B6%20-%20Gradio%20-%20%23fc7280" alt="Gradio">
                    <img src="https://img.shields.io/badge/%F0%9F%A4%96%20-%20Agentic%20AI%20-%20%23472731" alt="Agentic AI">
                    <img src="https://img.shields.io/badge/%F0%9F%A7%AE%20-%201B%2F2B%2F3B%20models%20-%20teal" alt="1B-3B Models">
                    <img src="https://img.shields.io/badge/%F0%9F%93%9D%20-%20Evaluation%20Report%20-%20purple" alt="Evaluation Report">
                    <img src="https://img.shields.io/badge/%F0%9F%93%8C%20-%20README-%20maroon?link=https%3A%2F%2Fhuggingface.co%2Fspaces%2FMCP-1st-Birthday%2FFistalAI%2Fblob%2Fmain%2FREADME.md" alt="ReadMe">
                </div>
            </div>
        """)

        
        gr.HTML("""
                
  <div style="text-align: left; color: #fff; line-height: 1.8;">
    
    <!-- Main header -->
    <div style="margin-bottom: 10px; 
                background: none !important; 
                padding: 15px; border-radius: 10px;">
        <h3 style="font-size: 1.4rem; margin-top: -10px; margin-left: -10px; font-size: 25px; color: #fff;">
           How does Fistal AI revolutionize LLM fine-tuning?
        </h3>
    </div>
    
    <div style="margin-bottom: 15px; background: linear-gradient(90deg, rgba(102,126,234,0.3), rgba(106,26,180,0.3), rgba(245,108,64,0.3)); 
                padding: 10px; border-radius: 10px; border-left: 4px solid #667eea;">
<strong style="color: #ada4f5; font-size: 1.1rem; 
               ">
    🧩 MCP + Agentic AI :
</strong>
<span style="font-size: 0.95rem; color: #e0e0e0;">
    Automates workflows using MCP while running autonomous data and training pipelines through Agentic AI.
</span>
    </div>

    <!-- Unsloth -->
    <div style="margin-bottom: 15px; background: linear-gradient(90deg, rgba(102,126,234,0.3), rgba(106,26,180,0.3), rgba(245,108,64,0.3)); 
                padding: 10px; border-radius: 10px; border-left: 4px solid #667eea;">
        <strong style="color: #ada4f5; font-size: 1.1rem;">🦥 Unsloth :</strong>
        <span style="font-size: 0.95rem; color: #e0e0e0;">
            Speeds up training with optimized kernels and memory-efficient 4-bit fine-tuning.
        </span>
    </div>

    <!-- Modal Labs -->
    <div style="margin-bottom: 15px; background: linear-gradient(90deg, rgba(102,126,234,0.3), rgba(106,26,180,0.3), rgba(245,108,64,0.3)); 
                padding: 10px; border-radius: 10px; border-left: 4px solid #667eea;">
        <strong style="color: #ada4f5; font-size: 1.1rem;">⚡ Modal Labs (with Volumes) :</strong>
        <span style="font-size: 0.95rem; color: #e0e0e0;">
            Provides serverless GPU compute with persistent volumes for fast scaling and reproducible experiments.
        </span>
    </div>

    <!-- Hugging Face -->
    <div style="margin-bottom: 15px; background: linear-gradient(90deg, rgba(102,126,234,0.3), rgba(106,26,180,0.3), rgba(245,108,64,0.3)); 
                padding: 10px; border-radius: 10px; border-left: 4px solid #667eea;">
        <strong style="color: #ada4f5; font-size: 1.1rem;">💾 Hugging Face + Gradio  :</strong>
        <span style="font-size: 0.95rem; color: #e0e0e0;">
            Model stored securely in HF repositories, Gradio's polished UI makes Fistal AI better.
        </span>
    </div>

    <!-- Gemini -->
    <div style="background: linear-gradient(90deg, rgba(102,126,234,0.3), rgba(106,26,180,0.3), rgba(245,108,64,0.3)); 
                padding: 10px; border-radius: 10px; border-left: 4px solid #667eea;">
        <strong style="color: #ada4f5; font-size: 1.1rem;">🔑 Gemini API Key :</strong>
        <span style="font-size: 0.95rem; color: #e0e0e0;">
            Enables secure access to the Gemini API that performs model orchestration and automated workflows.
        </span>
    </div>

</div>

""")
        gr.HTML("""
<div style="
    width: 100%;
    margin-top: 20px;
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(90deg, rgba(102,126,234,0.6), rgba(106,26,180,0.6), rgba(245,108,64,0.6));
    border-left: 5px solid #667eea;
    color: #fff;
    line-height: 1.6;
">

    <h2 style="
        margin: 0 0 12px 0;
        font-size: 1.4rem;
        color: #ada4f5;
        font-weight: 600;
        text-shadow: 0 0 1px #ada4f5;
    ">
        🚀 Start Fine-Tuning Now
    </h2>

    <p style="font-size: 0.95rem; color: #e0e0e0; margin-bottom: 10px;">
        Add your <strong style="color:white;">dataset topic</strong>, <strong style="color:white;">task type</strong>, number of samples, and your preferred <strong style="color:white;">Unsloth model</strong>.
    </p>

    <p style="font-size: 0.95rem; color: #e0e0e0;">
        Then sit back and watch <strong style="color:white;">Fistal AI</strong> automatically build datasets, fine-tune your LLMs, and deliver results like magic.
    </p>
                <p style="font-size: 0.95rem; color: #e0e0e0;"><i style="color: #e0e0e0;">
        Note: The process may take 30-45 minutes, depending on the number of samples and model chosen.</i>
    </p>
                

</div>

  

""")
        with gr.Group():
            with gr.Row():
                topic = gr.Textbox(label="📚 Dataset topic", placeholder="Python Questions, Return policy FAQS...")
                samples = gr.Slider(label="📊 Number of samples", minimum=0, maximum=2000, step=5, value=1000)
                task_type = gr.Dropdown(
                    label="🎯 Task Type", 
                    choices=["text-generation", "summarization", "classification", "question-answering"], elem_classes="drop"
                )
                model_name = gr.Dropdown(
                    label="🤖 Model to Fine-tune",
                    choices=[
                            "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
                            "unsloth/Phi-3-mini-4k-instruct",
                            "unsloth/Phi-3-medium-4k-instruct",
                            "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                            "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                            "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                            "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
                            "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
                            "unsloth/gemma-2-2b-it-bnb-4bit",
                            "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
                            "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
                            "unsloth/Granite-3.0-2b-instruct-bnb-4bit",
                            "unsloth/granite-4.0-h-1b-bnb-4bit"
                        ], elem_classes="drop"
                )
            
            tuner = gr.Button("🚀 Start Finetuning", size="lg", elem_id="tuner")
            
            gr.Markdown("""## 🔀 <span style="color: white;">Agent Activity Flow</span>""", elem_id="flow")
            status = gr.Textbox(label="Status", value="Ready to start...", interactive=False, elem_id="stat")
            
            with gr.Group(elem_classes="log-container"):
                output = gr.Markdown(label="Output Log:", value="", elem_classes="out")
                eval_report_storage = gr.Textbox(visible=False)  # Hidden storage
                copy_btn = gr.Button("Finetuning completed 🚀", visible=False, elem_id="copy-btn")
            
            

            async def run_workflow(dataset_topic, samples, model, task):
                output_log = "## Under the Hood\n\n"
                output_log += "📋 **Configuration:**\n\n"
                output_log += f"  • Topic: {dataset_topic}\n\n"
                output_log += f"  • Samples: {samples}\n\n"
                output_log += f"  • Model: {model}\n\n"
                output_log += f"  • Task: {task}\n\n"

                yield ("Starting workflow...", output_log, "", gr.Button(visible=False))

                try:
                    in_eval_report = False
                    eval_report_buffer = ""

                    async for chunk in run_fistal(
                        dataset_topic=dataset_topic,
                        num_samples=samples,
                        model_name=model,
                        task_type=task
                    ):
                        if "evaluating" in str(chunk).lower() or "llm_as_judge" in str(chunk).lower():
                            in_eval_report = True
                        
                        if in_eval_report:
                            eval_report_buffer += str(chunk)
                        else:
                            
                            output_log += str(chunk)

                        import re
                        urls = re.findall(r'https://huggingface\.co/[^\s\)]+', output_log + eval_report_buffer)
                        model_url = urls[0] if urls else ""
                        model_url = model_url.rstrip('.') 
                        model_url = re.sub(r'[^a-zA-Z0-9:/._-].*$', '', model_url)
                        
                        

                        yield ("🟡 Processing...", output_log + eval_report_buffer, eval_report_buffer, gr.Button(visible=False))
                        await asyncio.sleep(0.1)
                    

                    final_output = output_log
                    if eval_report_buffer:
                        final_output += "📊 **EVALUATION REPORT**\n\n"
                        final_output += eval_report_buffer

                    final_output += "\n\n✨ **Fistal AI has completed the process!**"
                    
                    yield ("🟢 Complete!", final_output, eval_report_buffer, gr.Button(visible=True))

                except Exception as e:
                    import traceback
                    error_log = output_log + f"\n\n❌ **ERROR:**\n```\n{str(e)}\n{traceback.format_exc()}\n```"
                    yield ("🔴 Error", error_log, "",gr.Button(visible=False))

            tuner.click(
                run_workflow, 
                [topic, samples, model_name, task_type], 
                [status, output, eval_report_storage, copy_btn]
            )
            def copy_report(report_text):
                if report_text and report_text.strip():
                    return report_text
                return ""

            copy_btn.click(
                copy_report,
                inputs=[eval_report_storage],
                outputs=[],
                js="""
                (text) => {
                    if (text && text.trim().length > 0) {
                        navigator.clipboard.writeText(text);
                        alert('✅ Evaluation report copied to clipboard!');
                    } else {
                        alert('Fistal has completed the process.');
                    }
                }
                """
            )

    return demo


    


if __name__ == "__main__":
    app().launch(theme= gr.themes.Ocean(), css=css)