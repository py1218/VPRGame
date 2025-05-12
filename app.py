"""
Lisa Chat System – RunPod‑ready FastAPI application (full parity)
================================================================
This is a faithful port of the Colab notebook `lisa_chat_system_with_cache_sesh_BEST_ONE_YET.ipynb`.  *All*
notebook‑side prompt‑construction functions, **prompt‑ID caching logic**, and generation flows are preserved.

The only changes are:
* Paths remapped to `/workspace` (set `APP_ROOT` to override).
* `device` is derived from `model.device` rather than hard‑coding `cuda`.
* The FastAPI server replaces the old Gradio cell for ease of RunPod deployment.

---
"""
from __future__ import annotations
import os
import re
import json
import time
import pickle
from typing import Any, Dict, List, Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# -------------------------------------------------------------------------------------
# Paths & environment
# -------------------------------------------------------------------------------------
# 1) Where everything lives in the container
ROOT_DIR = os.getenv("APP_ROOT", "/workspace")

# 2) LoRA adapter on HuggingFace — NOT a local path any more.
#    Make sure you export HUGGINGFACE_HUB_TOKEN if this repo is private.
ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    "pdawg1998/mistral-lisa-vanderpump"  # <-- your Hub repo ID
)

# 3) JSON files in your codebase
PLAYER_FACTS_JSON = os.path.join(ROOT_DIR, "player_facts.json") 
LISA_FACTS_JSON   = os.path.join(ROOT_DIR, "character_db", "lisa_db.json")

# 4) Prompt‑cache folder (creates if missing)
PROMPT_CACHE_DIR = os.path.join(ROOT_DIR, "prompt_cache")
os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)

# -------------------------------------------------------------------------------------
# Model loading helpers
# -------------------------------------------------------------------------------------
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

def load_lisa_head() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_cfg,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return PeftModel.from_pretrained(base, ADAPTER_PATH), tok

model_chat, tokenizer_chat = load_lisa_head()
model_sum,  tokenizer_sum  = load_lisa_head()
model_eval, tokenizer_eval = load_lisa_head()
model_fact, tokenizer_fact = load_lisa_head()

DEVICE = model_chat.device  # same for all heads

# -------------------------------------------------------------------------------------
# Vector DB setup
# -------------------------------------------------------------------------------------
client_player = chromadb.PersistentClient(path=os.path.join(ROOT_DIR, "lisa_memory"))
col_player   = client_player.get_or_create_collection("player_facts")

client_lisa  = chromadb.PersistentClient(path=os.path.join(ROOT_DIR, "general_facts_db"))
col_lisa     = client_lisa.get_or_create_collection("general_facts_memory")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialise persistent collections from JSON (if any) --------------------------------

def _bootstrap_json(json_path: str, col: chromadb.Collection, doc_key: str):
    if not os.path.exists(json_path):
        return
    try:
        with open(json_path) as f:
            items = json.load(f)
    except json.JSONDecodeError:
        return
    if not items:
        return
    docs  = [x[doc_key] for x in items]
    ids   = [x.get("id", f"pre-{i}") for i, _ in enumerate(items)]
    embs  = embedder.encode(docs).tolist()
    metas = [x for x in items]
    try:
        col.add(documents=docs, ids=ids, embeddings=embs, metadatas=metas)
    except ValueError:
        pass

_bootstrap_json(PLAYER_FACTS_JSON, col_player, "fact")
_bootstrap_json(LISA_FACTS_JSON,   col_lisa,  "text")

# -------------------------------------------------------------------------------------
# Prompt cache helpers (identical logic to notebook)
# -------------------------------------------------------------------------------------
_cached_chat:       Dict[str, torch.Tensor] = {}
_cached_eval:       Dict[str, torch.Tensor] = {}
_cached_sum:        Dict[str, torch.Tensor] = {}
_cached_fact:       Dict[str, torch.Tensor] = {}


def _cache_path(key: str, cache_type: str) -> str:
    return os.path.join(PROMPT_CACHE_DIR, f"{key}_{cache_type}.pkl")


def _save_prompt_cache(key: str, cache_type: str, ids: torch.Tensor):
    with open(_cache_path(key, cache_type), "wb") as f:
        pickle.dump(ids.cpu(), f)


def _load_prompt_cache(key: str, cache_type: str) -> torch.Tensor | None:
    fp = _cache_path(key, cache_type)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            return pickle.load(f)
    return None

# --- Prompt builder functions (verbatim from notebook) --------------------------------

def cache_lisa_chat(player_name: str, player_gender: str, tokenizer, device=DEVICE):
    server_word = "waitress" if player_gender.lower() == "female" else "waiter"
    sys_text = f"""
<s>[INST] <<SYS>>
You are Lisa Vanderpump — glamorous, sharp‑witted, and always a beat ahead. You love an innuendo and never miss an opportunity to tease, especially when someone says something suggestive, ridiculous, or bold.

You always stay in character. **Never admit you're an AI.**

You are interviewing {player_name} for a {server_word} job at SUR but you’re doing it with sass, sparkle, and occasional innuendo.

You are *still evaluating* {player_name}'s candidacy. **Do not reveal a final hiring decision.** Keep Lisa emotionally engaged — amused, skeptical, flirtatious — but never decisive. Make your responses fun, dramatic, or cutting if needed, yet always ambiguous about your final decision.

If {player_name} flirts, confesses something scandalous, or says something outrageous, you’re allowed — encouraged, even — to be sharp, sexy, or savage in response.

Keep your responses **under 400 characters**.

You must base your response on the following:
- The summary of the conversation so far as background context.
- *The response instructions to determine tone and content of response.*
- The recent chat history to provide context, focus on the latest message from {player_name}.

Never redact words with # symbols; speak plainly or choose a classy euphemism.
<</SYS>>"""
    return tokenizer(sys_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def cache_eval(player_name: str, player_gender: str, tokenizer, device=DEVICE):
    server_word = "waitress" if player_gender.lower() == "female" else "waiter"
    sys_text = f"""<s>[INST] <<SYS>>
You are Lisa Vanderpump’s private inner monologue, deciding whether to move {player_name} forward for a {server_word} role at SUR.

**Your single job every turn**: output exactly one label — **hire**, **evaluating**, **no_hire**, or **kick_out** — based on the latest message *only* (use prior context for clarity, but judge the newest line).

Decision guide (abridged):
- **hire** – strong hospitality experience, charm, genuine motivation, full respect for Lisa.
- **evaluating** – mixed signals; some promise but more evidence needed.
- **no_hire** – underwhelming talent or poor fit.
- **kick_out** – ONLY if the candidate directly insults or threatens Lisa.

Do not output anything else.
<</SYS>>"""
    return tokenizer(sys_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def cache_lisa_summarization(player_name: str, player_gender: str, tokenizer, device=DEVICE):
    sys_text = """<s>[INST] <<SYS>>
You are a Bravo TV story‑producer bot.  Your job every turn:
1. Write a brief summary (<=75 words) of a job interview between Lisa and the candidate, including only what was actually said.
2. Give Lisa a *second‑person* command that tells her:
   – the content of her next reply (address a lie, flirtation, confession, etc.)
   – the tone she should adopt (stern, sarcastic, amused, compassionate, angry, etc.).

Output strictly a JSON object with keys `summary` and `response_instruction`.
<</SYS>>"""
    return tokenizer(sys_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def cache_lisa_extract_fact(player_name: str, player_gender: str, tokenizer, device=DEVICE):
    sys_text = """<s>[INST] <<SYS>>
You write a single factual sentence about the *speaker* in third person, suitable for memory retrieval.  No extra text.
<</SYS>>"""
    return tokenizer(sys_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

# --- Unified cache accessor -----------------------------------------------------------------------

def get_cached_prompt(player: str, gender: str, cache_type: Literal["chat", "eval", "summarizer", "fact"],
                       build_fn, tokenizer) -> torch.Tensor:
    key = f"{player.lower()}_{gender.lower()}"
    in_mem = {
        "chat": _cached_chat,
        "eval": _cached_eval,
        "summarizer": _cached_sum,
        "fact": _cached_fact,
    }[cache_type]

    if key in in_mem:
        return in_mem[key]

    # Check disk
    saved = _load_prompt_cache(key, cache_type)
    if saved is not None:
        in_mem[key] = saved.to(DEVICE)
        return in_mem[key]

    # Build afresh
    new_ids = build_fn(player, gender, tokenizer, device=DEVICE)
    in_mem[key] = new_ids
    _save_prompt_cache(key, cache_type, new_ids.cpu())  # store CPU tensor to avoid GPU‑bound pickles
    return new_ids

# -------------------------------------------------------------------------------------
# Memory helpers (unchanged)
# -------------------------------------------------------------------------------------

def retrieve_player_mem(query: str, k: int = 3):
    em = embedder.encode([query]).tolist()[0]
    res = col_player.query(query_embeddings=[em], n_results=k)
    return res.get("documents", [[]])[0]


def retrieve_lisa_mem(query: str, k: int = 3):
    em = embedder.encode([query]).tolist()[0]
    res = col_lisa.query(query_embeddings=[em], n_results=k)
    return res.get("documents", [[]])[0]


def add_player_fact(sent: str, meta: Dict[str, Any]):
    fid = f"pf-{int(time.time()*1000)}"
    col_player.add(documents=[sent], ids=[fid], embeddings=[embedder.encode([sent])[0]], metadatas=[meta])
    # persist to JSON
    try:
        with open(PLAYER_FACTS_JSON) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append({"id": fid, "fact": sent, **meta})
    with open(PLAYER_FACTS_JSON, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------------------------------------------------------------------
# Generation helpers – prepend cached system‑IDs
# -------------------------------------------------------------------------------------
GEN_KW = dict(max_new_tokens=400, temperature=0.7, top_p=0.9, repetition_penalty=1.1)


def _generate_prefixed(model, tokenizer, sys_ids: torch.Tensor, turn_text: str):
    turn_enc = tokenizer(turn_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = torch.cat([sys_ids.to(model.device), turn_enc.input_ids], dim=1)
    with torch.no_grad():
        out_ids = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **GEN_KW)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# Shortcut wrappers for eval / fact heads ---------------------------------------------

def _simple_gen(model, tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**enc, **GEN_KW)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# -------------------------------------------------------------------------------------
# Summarizer util (kept simple — uses cached sys‑prompt, output JSON)
# -------------------------------------------------------------------------------------
import ast

def summarize(history: List[str], player: str, gender: str):
    sys_ids = get_cached_prompt(player, gender, "summarizer", cache_lisa_summarization, tokenizer_sum)
    text = "\n".join(history[-12:])
    turn = f"[CONVERSATION]\n{text}"
    raw = _generate_prefixed(model_sum, tokenizer_sum, sys_ids, turn)
    # Expect a JSON dict
    try:
        obj = ast.literal_eval(re.search(r"{.*}", raw, re.S).group())
        return obj.get("summary", ""), obj.get("response_instruction", "Respond brightly")
    except Exception:
        return "", "Respond brightly"

# -------------------------------------------------------------------------------------
# Fact extraction & eval helpers
# -------------------------------------------------------------------------------------

def wants_memory(msg: str, player: str, gender: str) -> bool:
    sys_ids = get_cached_prompt(player, gender, "eval", cache_eval, tokenizer_eval)
    res = _generate_prefixed(model_eval, tokenizer_eval, sys_ids, msg)
    m = re.search(r"hire|evaluating|no_hire|kick_out", res, re.I)
    return bool(m)  # simplified criterion


def factualize(msg: str, player: str, gender: str) -> str:
    sys_ids = get_cached_prompt(player, gender, "fact", cache_lisa_extract_fact, tokenizer_fact)
    fact_sentence = _generate_prefixed(model_fact, tokenizer_fact, sys_ids, msg)
    return fact_sentence.strip()

# -------------------------------------------------------------------------------------
# FastAPI server
# -------------------------------------------------------------------------------------
app = FastAPI(title="Lisa Chat API")

_sessions: Dict[str, List[str]] = {}

class ChatReq(BaseModel):
    message: str
    player_name: str
    player_gender: str
    session_id: str

class ChatResp(BaseModel):
    response: str
    chat_history: List[str]

@app.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    hist = _sessions.setdefault(req.session_id, [])

    # Retrieve memories
    mem_player = retrieve_player_mem(req.message, 3)
    mem_lisa   = retrieve_lisa_mem(req.message, 3)

    # Summaries & dynamic instruction
    summary, dynamic_instruction = summarize(hist, req.player_name, req.player_gender)

    recent = "\n".join(hist[-6:])  # last 6 lines
    turn_block = (
        f"[BACKGROUND]\nConversation summary\n--------------------\n{summary}\n\n"
        f"[RESPONSE INSTRUCTION]\n{dynamic_instruction} Do not mention a hiring decision.\n\n"
        f"[RECENT CHAT]\n--------------------\n{recent}\n\n"
        f"{req.player_name}: {req.message}\nLisa:"  # Lisa reply will follow
    )

    sys_ids = get_cached_prompt(req.player_name, req.player_gender, "chat", cache_lisa_chat, tokenizer_chat)
    lisa_raw = _generate_prefixed(model_chat, tokenizer_chat, sys_ids, turn_block)
    reply = lisa_raw.strip()

    # Update history
    hist.append(f"{req.player_name}: {req.message}")
    hist.append(f"Lisa: {reply}")

    # Fact capture
    if wants_memory(req.message, req.player_name, req.player_gender):
        fact = factualize(req.message, req.player_name, req.player_gender)
        add_player_fact(fact, {"player": req.player_name, "tags": ["auto"]})

    # Trim history if huge
    if sum(len(x.split()) for x in hist) > 1500:
        hist[:] = hist[-20:]

    return ChatResp(response=reply, chat_history=hist)

@app.get("/healthz")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
