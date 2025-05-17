from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"  # Nebo 3B verze
LORA_PATH = "./outputs"  # Složka s LoRA adapterem
MAX_TOKENS = 200

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"<|user|>\n{req.user_input}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = response.split("<|assistant|>\n")[-1].strip()
    return {"response": reply}
 