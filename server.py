#بسم الله الرحمن الرحيم
from fastmcp import FastMCP
import asyncio
import json
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import nltk
import sys
from modal_tool import train_with_modal, app, upload_to_hf_from_volume, evaluate_model

mcp = FastMCP(name="FistalMCP")


groq = os.getenv("GROQ_API_KEY")
hf = os.getenv("HF_TOKEN")

if not groq:
    print("GROQ_API_KEY missing!", file=sys.stderr)
if not hf:
    print("HF Token not valid", file=sys.stderr)

gk1 = os.environ.get("GOOGLE_API_KEY_1")
gk2 = os.environ.get("GOOGLE_API_KEY_2")
gk3 = os.environ.get("GOOGLE_API_KEY_3")

GOOGLE_API_KEYS = [k for k in [gk1, gk2, gk3] if k]

if not GOOGLE_API_KEYS:
    print("No Google API keys found!", file=sys.stderr)



async def genBatch(topic: str, samples_per_batch: int, batch_num: int, api_key: str, task_type: str) -> list:
    """Generate one batch of samples using a single API key"""
    
    if not api_key or api_key == "YOUR_API_KEY":
        return []
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )

    prompt_template = """
You are an expert dataset generator.
Generate authentic, high-quality data on the topic: {topic} for task type: {task_type} using your knowledge.
Generate exactly {num} concise, varied, and high-quality samples.
Return a JSON list of objects, each with keys: instruction, input, and output.
Do not add extra texts, markdown, or code fences.
RESPONSE:
"""

    promptJSON = ChatPromptTemplate.from_template(prompt_template)
    chain = promptJSON | llm

    try:
        user_input = {
            "topic": topic,
            "num": samples_per_batch,
            "task_type": task_type
        }

        response = await asyncio.to_thread(chain.invoke, user_input)
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()
        data = json.loads(content)

        return data if isinstance(data, list) else [data]
    
    except json.JSONDecodeError as e:
        print(f"JSON decode error in batch {batch_num}: {e}")
        return []
    except Exception as e:
        print(f"Error in batch {batch_num}: {e}")
        return []


@mcp.tool()
async def generate_json_data(topic: str,  task_type: str, num_samples: int = 1000) -> str:
    """
    Generate a training dataset with instruction, input, and output fields.
    Uses parallel batching for efficiency. Can generate up to 2000 samples.
    
    Args:
        topic: The topic or theme for the dataset
        num_samples: Number of training examples to generate (recommended: 100-2000)
    
    Returns:
        JSON string with status, topic, total_samples, and data array
    """
    topic = str(topic).strip() if topic else ""
    task_type = str(task_type).strip() if task_type else "text-generation"
    
    try:
        num_samples = int(num_samples)
    except (ValueError, TypeError):
        num_samples = 100

    if not topic:
        return json.dumps({
            "status": "error",
            "message": "Topic cannot be empty"
        })
    if num_samples <= 0 or num_samples > 2000:
        num_samples = min(max(50, num_samples), 2000)

    
    valid_keys = [k for k in GOOGLE_API_KEYS if k and k.strip() and k != "YOUR_API_KEY"]
    if not valid_keys:
        return json.dumps({
            "status": "error",
            "message": "No valid Google API keys configured"
        })

    start_time = time.time()
    samples_per_batch = 50
    total_batches = (num_samples + samples_per_batch - 1) // samples_per_batch

    try:
        tasks = []

        for batch_num in range(total_batches):
            api_key = valid_keys[batch_num % len(valid_keys)]
            task = genBatch(
                topic=topic.strip(),
                samples_per_batch=samples_per_batch,
                batch_num=batch_num + 1,
                api_key=api_key,
                task_type=task_type.strip()
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_samples = []
        for batch_result in results:
            if isinstance(batch_result, Exception):
                continue
            if isinstance(batch_result, list):
                all_samples.extend(batch_result)

        all_samples = all_samples[:num_samples]
        end_time = time.time()
        gen_time = end_time - start_time

        return json.dumps({
            "status": "success",
            "topic": topic,
            "task_type": task_type,
            "total_samples": len(all_samples),
            "requested_samples": num_samples,
            "total_batches": total_batches,
            "generation_time_seconds": round(gen_time, 1),
            "generation_time_minutes": round(gen_time / 60, 2),
            "samples_per_second": round(len(all_samples) / gen_time, 2) if gen_time > 0 else 0,
            "data": all_samples
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error generating dataset: {str(e)}"
        })


@mcp.tool()
async def format_json(raw_data) -> str:  
    """
    Convert raw dataset to ChatML format for training
    
    Args:
        raw_data: List or JSON string of samples with instruction/input/output
    
    Returns:
        JSON string with status, num_samples, and formatted data
    """
    try:
        if isinstance(raw_data, list):
            data = raw_data
        elif isinstance(raw_data, str):
            parsed = json.loads(raw_data)
            if isinstance(parsed, dict) and "data" in parsed:
                data = parsed["data"]
            else:
                data = parsed
        elif isinstance(raw_data, dict) and "data" in raw_data:
            data = raw_data["data"]
        else:
            return json.dumps({
                "status": "error",
                "message": f"Unexpected input type: {type(raw_data).__name__}"
            })
        
        if not isinstance(data, list):
            return json.dumps({
                "status": "error",
                "message": "Data must be a list of samples"
            })

        # Convert to ChatML format
        converted = []
        for item in data:
            if not isinstance(item, dict):
                continue
            
            if 'instruction' not in item or 'output' not in item:
                continue

            user_msg = str(item['instruction'])
            if item.get('input'):
                user_msg += f"\n\n{item['input']}"
            
            converted.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": str(item['output'])}
                ]
            })

        if not converted:
            return json.dumps({
                "status": "error",
                "message": "No valid samples to format"
            })

        return json.dumps({
            "status": "success",
            "num_samples": len(converted),
            "data": converted,
            "message": f"✅ Formatted {len(converted)} samples"
        }, ensure_ascii=False)
        
    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": f"Formatting failed: {str(e)}",
            "traceback": traceback.format_exc()
        })
        


@mcp.tool()
async def finetune_model(formatted_data, model_name: str, topic: str, task_type: str) -> str:  
    """
    Fine-tune model on Modal GPU
    
    Args:
        formatted_data: List or JSON string with formatted training samples
        model_name: Base model to fine-tune
    
    Returns:
        JSON string with status, repo_id, model_url
    """
    model_name = str(model_name).strip()
    
    models = [
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
    ]
    
    if model_name not in models:
        return json.dumps({
            "status": "error",
            "message": f"Model not supported. Choose from: {', '.join(models[:3])}..."
        })
    
    try:
        if isinstance(formatted_data, list):
            training_data = formatted_data
        elif isinstance(formatted_data, str):
            parsed = json.loads(formatted_data)
            if isinstance(parsed, dict) and "data" in parsed:
                training_data = parsed["data"]
            else:
                training_data = parsed
        elif isinstance(formatted_data, dict) and "data" in formatted_data:
            training_data = formatted_data["data"]
        else:
            return json.dumps({
                "status": "error",
                "message": f"Unexpected input type: {type(formatted_data).__name__}"
            })
        
        if not isinstance(training_data, list) or not training_data:
            return json.dumps({
                "status": "error",
                "message": "No training samples provided"
            })
        
        jsonl_content = "\n".join([json.dumps(s, ensure_ascii=False) for s in training_data])

        with app.run():
            result = train_with_modal.remote(jsonl_content, model_name)
        
        if result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": "Training failed"
            })
        
        repoTemp = """
Generate a short repository name for an unsloth finetuned model based on {topic} and {task_type}. 
Use '_' instead of spaces. Only return the name without quotations.
"""
        repoPrompt = ChatPromptTemplate.from_template(repoTemp)
        llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        api_key=groq
    )

        chain = repoPrompt | llm

        inp = {
            "topic": topic,
            "task_type": task_type
        }

        repoName = await asyncio.to_thread(chain.invoke, inp)
        repoName = repoName.content.strip()


        
        with app.run():
            hf_result = upload_to_hf_from_volume.remote(
                result["volume_path"], 
                result["timestamp"],
                repoName
            )

        return json.dumps({
            "status": "success",
            "repo_id": str(hf_result["repo_id"]),
            "model_url": str(hf_result["model_url"]),
            "model_path": str(hf_result["repo_id"]),
            "num_samples": len(training_data),
            "message": f"✅ Model at {hf_result['model_url']}"
        })

    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": f"Training failed: {str(e)}",
            "traceback": traceback.format_exc()
        })


@mcp.tool()
async def llm_as_judge(repo_id:str, topic: str, task_type: str) -> dict:
    """Use LLM to judge model quality based on topic and task type"""
    import evaluate
    eval_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=groq
    )
    test_prompt_text = f"""Generate 3 test cases for evaluating a model fine-tuned strictly based on **{topic} for {task_type}**.
Return ONLY a JSON array with this exact format, no other text:
[{{"input": "test question 1", "expected_output": "expected answer 1"}}, {{"input": "test question 2", "expected_output": "expected answer 2"}}, {{"input": "test question 3", "expected_output": "expected answer 3"}}]"""
    try:
        text_responses = await eval_llm.ainvoke(test_prompt_text)
        response = text_responses.content.strip()
        response = response.replace("```json", "").replace("```", "").strip()
        import re
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            response = match.group(0)
        
        test_cases = json.loads(response)[:3]

        test_inputs = [case['input'] for case in test_cases]

        with app.run():
            ft_output = evaluate_model.remote(repo_id, test_inputs)

        outputs = []
        for i, case in enumerate(test_cases):
            outputs.append(
                {
                    "input": case['input'],
                    "expected_output": case['expected_output'],
                    "model_output": ft_output[i]

                }
            )
        #METRICS:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")

        predictions = [output['model_output'] for output in outputs]
        references = [[output['expected_output']] for output in outputs]
        
        bleu_score = bleu.compute(predictions=predictions, references=references)
        rouge_score = rouge.compute(predictions=predictions, references=references)
        additional_metrics = {}
        if task_type.lower() in ["classification", "question-answering"]:
            accuracy_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load("f1")
            
            predictions_binary = [1 if pred.strip().lower() == ref[0].strip().lower() else 0 
                                for pred, ref in zip(predictions, references)]
            references_binary = [1] * len(predictions_binary)  
            
            accuracy_score = accuracy_metric.compute(predictions=predictions_binary, references=references_binary)
            f1_score = f1_metric.compute(predictions=predictions_binary, references=references_binary, average="binary")
            
            additional_metrics["accuracy"] = accuracy_score["accuracy"]
            additional_metrics["f1_score"] = f1_score["f1"]
        eval_prompt_text = f"""You are evaluating a model fine-tuned using Unsloth on the topic "{topic}" for {task_type} tasks.

**Your Task:** Provide an accurate, positive markdown evaluation report focusing on the model's strengths and capabilities based on your judgement and metrics.

**Test Results:**

Test Cases:
{json.dumps(test_cases, indent=2)}

Model Outputs:
{json.dumps(outputs, indent=2)}

**Metrics**
- BLEU Score: {bleu_score['bleu']:.4f}
- ROUGE-L Score: {rouge_score['rougeL']:.4f}
{f"- Accuracy: {additional_metrics.get('accuracy', 0):.4f}" if task_type.lower() in ["classification", "question-answering"] else ""}
{f"- F1 Score: {additional_metrics.get('f1_score', 0):.4f}" if task_type.lower() in ["classification", "question-answering"] else ""}

**Report Structure:**

## 🎉 Evaluation Report

### 📊 Performance Overview
Create a comparison table with columns: Test Input | Expected Output | Model Output | ✅ Assessment

### 🚀 Metrics:
- Explain each evaluated metrics and categorize the performance based on average threshold
- Use percentages and numerical figures to stance yoir report

### 💪 Key Strengths adn Weaknesses
- Accuracy and relevance
- Response coherence
- Task-specific capabilities
- Language quality


### ✨ Conclusion
Summarize the model's overall performance and recommended use cases.


Now write the complete evaluation report following this structure. Be enthusiastic and highlight strengths! 🎉"""

        
        eval_response = await eval_llm.ainvoke(eval_prompt_text)
        
        return {
            "status": "success",
            "report": str(eval_response.content),
            "test_cases": test_cases,
            "model_outputs": outputs
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }
        

        




if __name__ == "__main__":
    mcp.run()

    
    





