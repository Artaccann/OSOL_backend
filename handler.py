# handler.py

from unsloth import FastLanguageModel
import torch
import os

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=os.environ.get("HF_MODEL_NAME"),
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)
model.eval()

def handler(event):
    prompt = event["input"].get("prompt", "")
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": response.split("<|assistant|>\n")[-1].strip()}
