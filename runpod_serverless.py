import runpod
import torch
from unsloth import FastLanguageModel
import os

print("🚀 OSOL backend initializing...")

# Debug: výpis proměnných prostředí
print("🧪 HF_MODEL_NAME:", os.environ.get("HF_MODEL_NAME"))
print("🧪 LORA_PATH:", os.environ.get("LORA_PATH"))
print("🧪 HUGGINGFACE_TOKEN:", os.environ.get("HUGGINGFACE_TOKEN"))

# Načti model
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.environ["HF_MODEL_NAME"],
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        token=os.environ["HUGGINGFACE_TOKEN"]
    )
    model.load_adapter(
        os.environ["LORA_PATH"],
        token=os.environ["HUGGINGFACE_TOKEN"]
    )
    model.eval()
    print("✅ Model loaded and ready.")

except Exception as e:
    print("❌ Error during model initialization:", e)
    model = None
    tokenizer = None

# RunPod handler
def handler(event):
    try:
        if model is None or tokenizer is None:
            raise RuntimeError("Model is not initialized.")

        prompt = event["input"]["prompt"]
        print("💬 Prompt received:", prompt)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = response.split("<|assistant|>")[-1].strip()
        print("🤖 Model reply:", reply)
        return {"output": reply}

    except Exception as e:
        print("❌ Handler error:", e)
        return {"output": f"[Error] {str(e)}"}

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
