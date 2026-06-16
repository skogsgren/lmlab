import asyncio
from datetime import datetime
import gc
import math
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import stanza
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import re


class Settings(BaseSettings):
    lm: str = "AI-Sweden-Models/gpt-sw3-126m"
    tokenizers: list[str] = Field(
        default_factory=lambda: [
            "AI-Sweden-Models/gpt-sw3-126m",
            "gpt2",
            "facebook/xglm-564M",
            "google/byt5-small",
        ]
    )
    max_tokens: int = 50
    max_length_lm: int = 250
    max_length_tok: int = 90
    top_k: int = 10
    idle_timeout: int = 15
    pos_lang: str = "sv"
    pos_model_dir: Path = "./sparv"
    pos_pretrain: str = "sv_talbanken.pretrain.pt"
    pos_model: str = "sv_talbanken_tagger.pt"
    subpath: str = ""

    class Config:
        env_prefix = "lmlab_"


class InferenceRequest(BaseModel):
    inp: str


class NextTokenInferenceRequest(BaseModel):
    inp: str
    temp: float


settings = Settings()
print(settings)
app = FastAPI(root_path=settings.subpath)

QUEUE_LOCK = asyncio.Lock()

_last_access = time.time()
_tok, _model, _tokenizers, _stanza_pipeline, _stanza_tags = None, None, None, None, None


async def ensure_loaded():
    global _tok, _model, _tokenizers, _stanza_pipeline, _stanza_tags, _last_access
    async with QUEUE_LOCK:
        original_access = time.time()
        _last_access = time.time()
        if _model is None:
            print(f"{datetime.now().isoformat()} initializing models...")
            _tok = AutoTokenizer.from_pretrained(settings.lm)
            _model = AutoModelForCausalLM.from_pretrained(settings.lm).eval()
            _tokenizers = {}
            for tokenizer in settings.tokenizers:
                _tokenizers[tokenizer] = AutoTokenizer.from_pretrained(tokenizer)
            _stanza_pipeline = stanza.Pipeline(
                lang=settings.pos_lang,
                processors="tokenize,pos",
                dir=str(settings.pos_model_dir),
                pos_pretrain_path=str(settings.pos_model_dir / settings.pos_pretrain),
                pos_model_path=str(settings.pos_model_dir / settings.pos_model),
                download_method=None,
            )
            _stanza_tags = sorted(
                [
                    x
                    for x in _stanza_pipeline.processors["pos"]
                    .vocab["upos"]
                    ._unit2id.keys()
                    if x[0] != "<"
                ]
            )

            print(
                f"{datetime.now().isoformat()} finished initializing models (took {time.time() - original_access:01}s ..."
            )


async def unload_models():
    global _tok, _model, _tokenizers
    async with QUEUE_LOCK:
        print(f"{datetime.now().isoformat()} unloading models...")
        _tok, _model, _tokenizers = None, None, None
        _stanza_pipeline, _stanza_tags = None, None
        gc.collect()


async def idle_monitor():
    global _last_access, _model
    while True:
        await asyncio.sleep(60)
        idle = time.time() - _last_access
        if idle > (settings.idle_timeout * 60):
            if _model is not None:
                await unload_models()


@app.on_event("startup")
async def startup():
    asyncio.create_task(idle_monitor())


@app.post("/seq")
async def seq(payload: InferenceRequest):
    print(f"{datetime.now().isoformat()} received request for seq")
    await ensure_loaded()
    global _tok, _model

    async with QUEUE_LOCK:
        inp = payload.inp[: settings.max_length_lm]
        if inp == "":
            return {
                "tokens": [],
                "topk": [],
                "actual": [],
                "summary": {},
                "note": "field cannot be empty",
            }

        enc = _tok(inp, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        input_ids = torch.cat(
            [
                torch.tensor(
                    [_tok.bos_token_id or _tok.eos_token_id],
                    dtype=input_ids.dtype,
                ),
                input_ids,
                torch.tensor([_tok.eos_token_id], dtype=input_ids.dtype),
            ]
        )
        enc = {"input_ids": input_ids.unsqueeze(0)}
        out = _model(**enc, use_cache=False, return_dict=True)
        logits = out.logits[0]
        tokens = _tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())

        topk_all = []
        actual = []
        logprob_sum = 0.0
        N = len(input_ids) - 1
        for i in range(N):
            probs = torch.softmax(logits[i], dim=-1)
            topk_prob, topk_idx = torch.topk(probs, k=settings.top_k)
            topk = []
            for p, idx in zip(topk_prob.tolist(), topk_idx.tolist()):
                topk.append(
                    {"token": _tok.convert_ids_to_tokens([idx])[0], "prob": float(p)}
                )
            topk_all.append(topk)

            tid = int(input_ids[i + 1].item())
            p = float(probs[tid].item())
            lp = float(torch.log(probs[tid]).item())
            logprob_sum += lp
            actual.append({"token": tokens[i + 1], "prob": p, "logprob": lp, "pos": i})

        return {
            "tokens": tokens[1:-1],
            "topk": topk_all[:-1],
            "actual": actual[:-1],
            "summary": {
                "log_prob": logprob_sum,
                "log10_prob": logprob_sum / math.log(10),
                "avg_log_prob": logprob_sum / (N - 1),
                "num_predicted": N - 1,
            },
        }


@app.post("/next")
async def next(payload: NextTokenInferenceRequest):
    print(f"{datetime.now().isoformat()} received request for next")
    await ensure_loaded()
    global _tok, _model

    async with QUEUE_LOCK:
        enc = _tok(
            payload.inp,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        if enc["input_ids"].shape[1] == 0:
            input_ids = torch.tensor(
                [_tok.bos_token_id or _tok.eos_token_id], dtype=int
            )
            truncated = False
        else:
            input_ids = enc["input_ids"][0]
            truncated = enc["input_ids"].shape[1] >= settings.max_tokens
            input_ids = input_ids[: settings.max_tokens - 1]
            input_ids = torch.cat(
                [
                    torch.tensor(
                        [_tok.bos_token_id or _tok.eos_token_id],
                        dtype=input_ids.dtype,
                    ),
                    input_ids,
                ]
            )
        input_ids = input_ids.unsqueeze(0)
        enc = {"input_ids": input_ids}

        out = _model(**enc, use_cache=False, return_dict=True)
        logits = out.logits[0]
        probs = torch.softmax(logits[-1], dim=-1)
        tempered_probs = torch.softmax(logits[-1] / (payload.temp + 1e-6), dim=-1)
        topk_prob, topk_idx = torch.topk(probs, k=min(settings.top_k, probs.shape[-1]))
        tempered_topk_prob = tempered_probs[topk_idx]

        topk = []
        for p, tp, idx in zip(
            topk_prob.tolist(), tempered_topk_prob.tolist(), topk_idx.tolist()
        ):
            tok_str = _tok.convert_ids_to_tokens([idx])[0]
            topk.append(
                {
                    "token": tok_str,
                    "id": int(idx),
                    "prob": float(p),
                    "temp_prob": float(tp),
                }
            )
        return {
            "input": {
                "text": payload.inp,
                "temp": payload.temp,
                "truncated": truncated,
                "tokens": [
                    t for t in _tok.convert_ids_to_tokens(input_ids[0].tolist())
                ][1:],
            },
            "topk": topk,
        }


@app.post("/tok")
async def tokenizers(payload: InferenceRequest):
    await ensure_loaded()
    global _tokenizers
    print(f"{datetime.now().isoformat()} received request for tokenizers")
    async with QUEUE_LOCK:
        inp = payload.inp[: settings.max_length_tok]
        results = {}
        for spec, tok in _tokenizers.items():
            tokens = tok(inp, add_special_tokens=False)["input_ids"]
            results[spec] = [(tok.decode(x), x) for x in tokens]
        return results


@app.post("/pos")
async def pos(payload: InferenceRequest):
    await ensure_loaded()
    global _stanza_pipeline, _stanza_tags
    async with QUEUE_LOCK:
        inp = payload.inp[: settings.max_length_lm]
        return {
            "inp": inp,
            "tags": _stanza_tags,
            "words": [
                (word.text, word.pos)
                for sentence in _stanza_pipeline(inp).sentences
                for word in sentence.words
            ],
        }


@app.get("/")
async def read_root(request: Request):
    global settings
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "max_length_tok": settings.max_length_tok,
            "max_length_lm": settings.max_length_lm,
            "max_tokens": settings.max_tokens,
        },
    )


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/dist", StaticFiles(directory=BASE_DIR / "dist"), name="dist")
