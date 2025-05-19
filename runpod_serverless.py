import runpod
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch
from unsloth import FastLanguageModel
import os

# Načti model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=os.environ["HF_MODEL_NAME"],
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    token=os.environ["HUGGINGFACE_TOKEN"]
)
model.load_adapter(os.environ["LORA_PATH"], token=os.environ["HUGGINGFACE_TOKEN"])
model.eval()

# RunPod handler funkce
def handler(event):
    prompt = event["input"]["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=250)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return { "output": response }

# registruj handler
runpod.serverless.start({"handler": handler})
