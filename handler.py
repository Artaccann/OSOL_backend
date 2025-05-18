from unsloth import FastLanguageModel
import torch, os, json

# ⏬ Načtení modelu
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=os.environ.get("HF_MODEL_NAME"),
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)
model.eval()

# ⏬ Hlavní handler
def handler(event):
    prompt = event["input"].get("prompt", "")
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = response.split("<|assistant|>\n")[-1].strip()

    # 🧠 Pokus o parsování výstupu jako JSON
    try:
        parsed = json.loads(reply)
        return parsed  # Pokud je to validní JSON → pošli jak je
    except:
        return {"output": reply}  # Jinak zabal do očekávaného formátu
