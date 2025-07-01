# api.py
from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI()

# Load model and tokenizer from Hugging Face (Mistral)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Optional API token authentication
def verify_token(request: Request):
    api_token = os.getenv("API_TOKEN")
    auth_header = request.headers.get("Authorization")
    if api_token and auth_header != f"Bearer {api_token}":
        raise HTTPException(status_code=403, detail="Unauthorized")

@app.get("/")
def root():
    return {"message": "Mistral API is running"}

@app.post("/chat")
async def chat(request: Request):
    verify_token(request)
    body = await request.json()
    prompt = body.get("prompt", "Hello!")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
