from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch
import os
from unsloth import FastLanguageModel
from dotenv import load_dotenv

load_dotenv()

# Konfigurace
base_model_id = os.getenv("HF_MODEL_NAME", "unsloth/Llama-3.2-1B")
lora_path = os.getenv("LORA_PATH", "Artaccann/OSOL_LoRA_v1")

# Načti model
print("⏳ Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    token=os.getenv("HUGGINGFACE_TOKEN")
)

# Aplikuj LoRA adaptér
model.load_adapter(lora_path, token=os.getenv("HUGGINGFACE_TOKEN"))
model.eval()
print("✅ Model ready!")

# FastAPI server
app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=250)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse(content={"response": response})
