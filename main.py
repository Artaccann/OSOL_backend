from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from unsloth import FastLanguageModel
import torch
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ⏬ Nastavení
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "Artaccann/OSOL_backend")
MAX_TOKENS = 200

# ⏬ Načtení modelu + tokenizeru z HF
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)
model.eval()

# ⏬ Pydantic schéma
class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"<|user|>\n{req.user_input}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = response.split("<|assistant|>\n")[-1].strip()

    # 🧠 Pokus o parsování JSON odpovědi
    try:
        parsed = json.loads(reply)
        return JSONResponse(content=parsed)
    except:
        return {"output": reply}
