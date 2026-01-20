<div align= "center">
<h1>🚀 Fistal AI - Autonomous Fine-Tuning Platform </h1>
</div>

<div align="center">

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20-%20HF%20Space%20-%20orange)](https://huggingface.co/spaces/your-username/fistal-ai)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Modal](https://img.shields.io/badge/Modal-Enabled-green)
![Gemini](https://img.shields.io/badge/%E2%9C%A8%20-%20Gemini%20API%20-%20teal)
![Unsloth](https://img.shields.io/badge/Unsloth-4bit-purple)
![MCP](https://img.shields.io/badge/MCP-Enabled-pink)
![Gradio](https://img.shields.io/badge/%F0%9F%94%B6%20-%20Gradio%20-%20%23fc7280)
![Agentic AI](https://img.shields.io/badge/%F0%9F%A4%96%20-%20Agentic%20AI%20-%20%23472731)
![1B-3B Models](http14903s://img.shields.io/badge/%F0%9F%A7%AE%20-%201B%2F2B%2F3B%20models%20-%20teal)
![Evaluation Report](https://img.shields.io/badge/%F0%9F%93%9D%20-%20Evaluation%20Report%20-%20purple)

**Agentic AI that seamlessly finetunes LLM's with Unsloth and Modal**

[🎮 Try Demo](https://drive.google.com/file/d/1-Uf2-k-gJsIozg-YX0oo_qWjeS31sq98/view?usp=sharing) • [📱 LinkedIn Post](https://www.linkedin.com/posts/mahreen-fathima-anis-5238ba36b_fistal-ai-a-hugging-face-space-by-mcp-1st-birthday-activity-7400939406448074752-SKAV?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFvK0WsBW7LU9mIHS4nf2zGkEQ85Wi322Sg) 

</div>

---

## 🎯 What is Fistal AI?

Fistal AI is an **autonomous fine-tuning platform** that transforms the complex process of training custom language models into a single-click experience. Simply specify your topic, and Fistal handles everything:

- 🤖 **Synthetic Dataset Generation** - Creates high-quality training data using LLMs
- 🔄 **Automatic Data Formatting** - Converts to chat/instruction format
- 🏋️ **Serverless Training** - Fine-tunes models on Modal's GPU infrastructure
- 📊 **LLM-as-Judge Evaluation** - Validates model performance
- 🤗 **Hugging Face Deployment** - Publishes your model automatically

**No ML expertise required. No infrastructure setup. Just results.**

---

## ✨ Features

### 🎨 **Intuitive Interface**
- Clean Gradio-based web UI hosted on Hugging Face Spaces
- Real-time training progress with educational insights
- Automatic Hugging Face integration with one-click model access
- Direct model upload in native HF format (ready to use immediately)

### ⚡ **Blazing Fast Training**
- **3x faster dataset generation** with parallel API calls (Gemini)
- **2x faster training** with Unsloth optimization and Modal GPU's
- **70% less memory** usage via 4-bit quantization
- Training completes in **10-20 minutes** for 500 samples

### 🧠 **Smart Defaults**
- 4-bit quantization for optimal quality/size balance
- LoRA fine-tuning (updates only 0.1% of parameters)
- Supports 1B-3B parameter models (Qwen, Llama, Gemma, Phi)
- Automatic hyperparameter optimization
- Native HF format upload (no conversion needed)

### 🔬 **Quality Assurance**
- LLM-as-judge evaluation system
- Coherence, relevance, and accuracy testing
- Comprehensive evaluation reports
- Real-time monitoring of training metrics

### 🔌 **MCP-Powered Workflow**
- Agentic orchestration using Model Context Protocol (MCP)
- 4 specialized MCP tools for end-to-end automation
- Intelligent decision-making throughout the pipeline
- Seamless tool coordination for optimal results

---
## ✨ Sponsors:

- **Modal Labs** : Seamless T4 GPU access
- **Gemini API** : Handles majority of LLM tasks including data generation and agentic control
---

## Watch Demo:

 *Note: The demo runs only 5 samples for speed, but you can scale it to 2000+ in real use.*

 [**Demo Fistal**](https://drive.google.com/file/d/1wXxGDKUfQXmntW3ldhy-rov8Kjs_K3dj/view?usp=sharing)

---

### **How Fistal AI Works Behind the Scenes**

* Fistal AI runs on an **agentic workflow** powered by LangGraph.
* Instead of a fixed script, an **AI agent** decides what step to run next.
* All the actual work (dataset generation, formatting, training, evaluation) is done by **MCP tools**.
* The agent just thinks → MCP tools do the work → agent continues automatically.

---


### **The MCP Server**

* Hosts four tools:

  * `generate_json_data` → creates synthetic training data
  * `format_json` → converts it to ChatML format
  * `finetune_model` → runs Unsloth training on Modal
  * `llm_as_judge` → evaluates the trained model
* Each tool is isolated and safe.
* Returns clean, structured results that the agent uses.

---

### **Pipeline Flow (Step-by-Step)**

* **1. Dataset Generation**
  Agent calls the tool → LLMs generate 20–500 examples in parallel.

* **2. Dataset Formatting**
  Agent calls next tool → raw dataset becomes ChatML/instruction format.

* **3. Fine-Tuning**
  Agent launches training on Modal using Unsloth + 4-bit QLoRA.

* **4. Evaluation**
  Agent runs LLM-as-judge → gets coherence/relevance/accuracy/ROUGE/BLEU scores with evaluate library.

* **5. Final Output**
  The model and adapters are automatically uploaded to the user's(mahreenfathima) Hugging Face account (based on the HF token provided).
  Automatic Evaluation Report generated.

  

---

#### 🛠️ **The 4 MCP Tools**

1. **`generate_json_data`**
   - **Purpose**: Synthetic dataset generation
   - **Input**: Topic, sample count, task type
   - **Process**: Parallel API calls to Gemini + Groq with intelligent prompt engineering
   - **Output**: JSON dataset with diverse, high-quality examples
   - **MCP Role**: Agent invokes this tool first, receives confirmation, then proceeds

2. **`format_json`**
   - **Purpose**: Convert raw data to training(ChatML) format
   - **Input**: Raw JSON dataset path
   - **Process**: Transforms to chat/instruction format optimized for fine-tuning
   - **Output**: Formatted dataset ready for training
   - **MCP Role**: Agent receives dataset path from previous tool, formats it automatically

3. **`finetune_model`**
   - **Purpose**: Execute serverless training
   - **Input**: Formatted dataset, model name, hyperparameters
   - **Process**: Deploys training job to Modal with Unsloth optimization
   - **Output**: Fine-tuned model weights + training metrics
   - **MCP Role**: Agent monitors training progress, handles failures, manages GPU resources
   - **Internal Functions** (executed within Modal):
     - `train_with_modal`: Runs finetuning process with Unsloth and saves model in Volume
     - `upload_to_hf_from_volume`: Pushes the trained model weights to Hugging Face Hub repository

4. **`llm_as_judge`**
   - **Purpose**: Quality evaluation
   - **Input**: Fine-tuned model path, test cases
   - **Process**: Generates test prompts, evaluates responses, scores quality
   - **Output**: Comprehensive evaluation report with metrics
   - **MCP Role**: Final validation step, agent parses results and presents to user
  - **Internal Functions** (executed within Modal):
     - `evaluate_model`: Runs validation metrics on the fine-tuned model during/after training
     

#### 🧠 **Fistal's Agentic Approach**

```python
# Agent makes decisions based on context
agent decides: "User wants Python dataset"
  → invokes generate_json_data with optimal parameters
  
agent observes: "Dataset generated successfully"
  → invokes format_json with received path
  
agent monitors: "Training at 50%, loss decreasing"
  → continues monitoring, adjusts if needed
  
agent validates: "Model trained, run evaluation"
  → invokes llm_as_judge for quality check
```

**Benefits**:
- 🎯 **Intelligent Decision Making**: Agent chooses best parameters and strategies
- 🔄 **Error Recovery**: Automatically retries failed steps with adjusted parameters
- 📊 **Context Awareness**: Each tool receives relevant context from previous steps
- 🔒 **Security**: MCP provides secure tool execution
- 🔧 **Modularity**: Tools can be updated independently without breaking the workflow
- 📈 **Scalability**: Easy to add new tools (e.g., hyperparameter tuning, multi-GPU training)


---

## 🛠️ Tech Stack

<table>
<tr>
<td width="50%">

### Core Technologies
- **[Unsloth](https://github.com/unslothai/unsloth)** - 2x faster training, 70% less VRAM
- **[Modal](https://modal.com)** - Serverless GPU infrastructure
- **[Gradio](https://gradio.app)** - Web interface on HF Spaces
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agentic workflow orchestration
- **[MCP](https://modelcontextprotocol.io)** - Tool integration protocol
- **[HUGGING FACE](https://huggingface.co/)** - Uploads model into repository with hf tokens

</td>
<td width="50%">

### AI Models & APIs
- **Gemini Flash 2.0** - Fast dataset generation
- **Groq (Llama 3.1 70B)** - LLM evaluation
- **Hugging Face** - Model hosting & deployment
- **4-bit Quantization** - Optimal quality/size balance
- **Native HF Upload** - No format conversion needed

</td>
</tr>
</table>

---



## 📊 Performance Metrics

| Metric | Value | Details |
|--------|-------|---------|
| **Dataset Generation** | 3x faster | Parallel processing with API keys |
| **Training Speed** | 2x faster | Unsloth optimization |
| **Memory Usage** | -70% | 4-bit quantization |
| **Training Time** | 10-20 min | For 500 samples on T4 GPU |
| **Model Size** | ~1-2 GB | Native HF format (safetensors) |
| **Parameters Updated** | 0.1% | LoRA efficiency |
| **MCP Tools** | 4 | Autonomous workflow management |

---

## 🔧 Supported Models & Tasks

### Prominent Models (1B-3B Parameters)
- `Qwen/Qwen2.5-1.5B-Instruct` 
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `google/gemma-2-2b-it`
- `microsoft/Phi-3.5-mini-instruct`

### Popular Task Types
- **text-generation**: General text completion and content creation
- **question-answering**: Q&A pairs and knowledge retrieval


### Output Format
- **Native Hugging Face format** (safetensors + adapter weights)
- Immediately usable with transformers library
- Compatible with HF Inference API

---

## 🎮 Try It Now

<div align="center">

### 🚀 **[Launch Fistal AI Demo](https://drive.google.com/file/d/1-Uf2-k-gJsIozg-YX0oo_qWjeS31sq98/view?usp=sharing)**

### 📱 **[Read LinkedIn Post](https://www.linkedin.com/posts/mahreen-fathima-anis-5238ba36b_fistal-ai-a-hugging-face-space-by-mcp-1st-birthday-activity-7400939406448074752-SKAV?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFvK0WsBW7LU9mIHS4nf2zGkEQ85Wi322Sg)**

**Hosted on Hugging Face Spaces - No installation required!**

</div>

---

## 📝 License

This project is licensed under the APACHE License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Anthropic MCP](https://modelcontextprotocol.io)** - For the powerful tool integration protocol
- **[Unsloth](https://github.com/unslothai/unsloth)** - For making fine-tuning accessible and fast
- **[Modal](https://modal.com)** - For serverless GPU infrastructure
- **[Hugging Face](https://huggingface.co)** - For model hosting and Spaces platform
- **[Google Gemini](https://ai.google.dev/)** - For powerful API access
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - For agentic orchestration framework
- **[Gradio](https://gradio.app)** - For building the interactive UI effortlessly 


---

<div align="center">



**Powered by MCP • Unsloth • Modal • Hugging Face • Gemini API**

❤️ Like our space our HuggingFace • 🚀 Try the demo • 📱 Share on LinkedIn

</div>










