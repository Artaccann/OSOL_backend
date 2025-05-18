# handler.py

from unsloth import FastLanguageModel
import torch, os

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=os.environ.get("HF_MODEL_NAME"),
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)
model.eval()

def handler(event):
    print("=== Handler was called ===")
    prompt = event.get("input", {}).get("prompt", "")
    print(f"Prompt: {prompt}")
    

    
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    print("DEBUG: Generation complete.")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"DEBUG: Raw response: {response}")
    
    final = response.split("<|assistant|>\n")[-1].strip()
    print(f"DEBUG: Final output: {final}")
    
    return {"output": final}

