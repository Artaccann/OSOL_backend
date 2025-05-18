import runpod

def handler(event):
    print("=== HANDLER TRIGGERED ===")
    prompt = event.get("input", {}).get("prompt", "")
    return {"output": f"Echo: {prompt}"}

runpod.serverless.start({"handler": handler})
