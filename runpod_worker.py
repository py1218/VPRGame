import os, time, json, re, pickle
import torch
import runpod
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ENV
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "pdawg1998/mistral-lisa-vanderpump")

ROOT_DIR = os.getenv("APP_ROOT", "/workspace")
PROMPT_CACHE_DIR = os.path.join(ROOT_DIR, "prompt_cache")
os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

def load_head():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        use_auth_token=HF_TOKEN,
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_auth_token=HF_TOKEN
    )
    tok.pad_token = tok.eos_token
    return PeftModel.from_pretrained(base, ADAPTER_PATH, use_auth_token=HF_TOKEN), tok

print("🔄 Loading Lisa model...")
model_chat, tokenizer_chat = load_head()
DEVICE = model_chat.device

GEN_KW = dict(max_new_tokens=200, temperature=0.7, top_p=0.9)

def build_prompt(player, msg):
    return f"[INST] <<SYS>>You are Lisa Vanderpump. Stay in character.<</SYS>>\n{player}: {msg} [/INST]"

def lisa_reply(player, msg):
    prompt = build_prompt(player, msg)
    enc = tokenizer_chat(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model_chat.generate(**enc, **GEN_KW)
    return tokenizer_chat.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

def handler(event: Dict[str, Any]):
    inp = event.get("input", {})
    msg = inp.get("prompt", "")
    player = inp.get("player_name", "Player")
    reply = lisa_reply(player, msg)
    return {"response": reply}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})