import modal
import json 
from datasets import Dataset
import time


modal.enable_output()

app = modal.App("fistalfinetuner")

volume = modal.Volume.from_name("fistal-models", create_if_missing=True )





modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu121",
        
    )
    .pip_install(
        "transformers",
        "datasets",
        "accelerate",
        "trl",
        "bitsandbytes",
        "peft",
        "unsloth_zoo",
        "datasets==4.3.0"
    )
    .pip_install(
        "unsloth @ git+https://github.com/unslothai/unsloth.git"
    )
)

@app.function(
    image=modal_image, 
    gpu="T4", 
    timeout=3600,
    volumes={"/models":volume},
    retries=modal.Retries(max_retries=0, backoff_coefficient=1.0)
)
def train_with_modal(ft_data: str, model_name: str):
    """
    Finetuning model using Modal's GPU
    """
    import torch
    
    if not torch.cuda.is_available():
        return {"status": "error", "message": "No GPU available!"}
    
    from unsloth import FastLanguageModel, is_bf16_supported
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import os

    data = []
    for line in ft_data.strip().split('\n'):
        if line.strip():
            data.append(json.loads(line))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        load_in_4bit=True,
        dtype=None
    )

    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=2001,
        use_gradient_checkpointing="unsloth",
        loftq_config=None,
        use_rslora=False
    )
    
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2000,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=1,
            max_steps=30,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            output_dir="/tmp/training_output",
            seed=42,
            report_to="none",
            dataloader_num_workers=0
        )
    )
    print("Training started...")
    trainer.train()
    print("Training complete!")

    timestamp = int(time.time())
    volume_path = f"/models/finetuned-{timestamp}"

    os.makedirs(volume_path, exist_ok=True)
    print(f"Saving to: {volume_path}")
    

    model.save_pretrained_merged(volume_path, tokenizer, save_method="merged_16bit")
    print("Model saved!")
    model.config.save_pretrained(volume_path)

    trainer.save_model(volume_path)
    tokenizer.save_pretrained(volume_path)





    volume.commit()
    print("Volume has been committed!")

    del model
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "status":"success",
        "volume_path":volume_path,
        "timestamp": timestamp

    }




@app.function(
    image=modal_image,
    volumes={"/models": volume},
    timeout=900,  
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def upload_to_hf_from_volume(volume_path: str, timestamp: int, repoName: str):
    """
    Upload model directly from Modal Volume to HuggingFace
    This runs on Modal's fast network - no download to local machine needed!
    """
    from huggingface_hub import HfApi, create_repo
    import os
    
    print(f"📤 Uploading from {volume_path} to HuggingFace...")
    
    if not os.path.exists(volume_path):
        raise FileNotFoundError(f"Model not found at: {volume_path}")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in Modal secrets")
    
    hf_api = HfApi()
    repo_id = f"mahreenfathima/finetuned-{repoName}-{timestamp}"
    
    print(f"Creating HuggingFace repo: {repo_id}")
    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=False,
        exist_ok=True,
        repo_type="model"
    )
    
    print(f"Uploading files to {repo_id}...")
    hf_api.upload_folder(
        folder_path=volume_path,
        repo_id=repo_id,
        token=hf_token,
        commit_message=f"Fine-tuned model (timestamp: {timestamp})"
    )
    
    model_url = f"https://huggingface.co/{repo_id}"
    print(f"✅ Successfully uploaded to {model_url}")
    
    return {
        "model_url": model_url,
        "repo_id": repo_id
    }

@app.function(
    gpu="T4",
    timeout=600,
    image=modal_image
)
def evaluate_model(repo_id: str, test_inputs: list[str]):
    """Load model and run inference on test cases"""
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    import torch
    
    print(f"Loading model: {repo_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(  
        model_name=repo_id,
        max_seq_length=512,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    
    outputs = []
    for test_input in test_inputs:
        print(f"Processing: {test_input[:50]}...")
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                do_sample=True
            )
        
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if decoded.startswith(test_input):
            decoded = decoded[len(test_input):].strip()
        outputs.append(decoded)
    
    return outputs


