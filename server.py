# vim: set expandtab tabstop=4 shiftwidth=4 softtabstop=4 belloff=all :

import asyncio
import math
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import re

GPT2_MODEL = "distilgpt2"
BERT_MODEL = "distilbert-base-uncased"
TOPK = 12
MAX_CHARS = 200

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Access-Control-Allow-Origin"],
)


class Inp(BaseModel):
    text: str


_tok_gpt = AutoTokenizer.from_pretrained(GPT2_MODEL)
_mdl_gpt = AutoModelForCausalLM.from_pretrained(GPT2_MODEL).eval()

cfg = AutoConfig.from_pretrained(BERT_MODEL)
cfg.attn_implementation = "eager"
_tok_bert = AutoTokenizer.from_pretrained(BERT_MODEL)
_mdl_bert = AutoModel.from_pretrained(BERT_MODEL, config=cfg).eval()

QUEUE_LOCK = asyncio.Lock()


def limit_chars(s: str, max_chars: int = MAX_CHARS):
    if not s:
        return "", 0, False
    if len(s) > max_chars:
        return s[:max_chars], len(s), True
    return s, len(s), False


@app.post("/gpt2")
async def gpt2_endpoint(inp: Inp):
    async with QUEUE_LOCK:
        with torch.inference_mode():
            text, count, was_truncated = limit_chars((inp.text or ""))
            if text == "":
                return {
                    "tokens": [],
                    "topk": [],
                    "actual": [],
                    "summary": {},
                    "note": "field cannot be empty",
                }

            def detok_single(token_str: str) -> str:
                """GPT2 has special tokens the user shouldn't see"""
                if token_str == "Ċ":
                    return "<new-line>"
                token_str = re.sub("Ġ", "\u2423", token_str)
                try:
                    return _tok_gpt.convert_tokens_to_string([token_str])
                except Exception:
                    return _tok_gpt.decode(_tok_gpt.convert_tokens_to_ids([token_str]))

            enc = _tok_gpt(text, return_tensors="pt", add_special_tokens=False)
            # concat BOS + input_ids since GPT2 doesn't do that
            input_ids = enc["input_ids"][0]
            input_ids = torch.cat(
                [
                    # EOS = BOS for GPT2 for some reason
                    # https://github.com/huggingface/transformers/issues/3311
                    torch.tensor(
                        [_tok_gpt.bos_token_id or _tok_gpt.eos_token_id],
                        dtype=input_ids.dtype,
                    ),
                    input_ids,
                    torch.tensor([_tok_gpt.eos_token_id], dtype=input_ids.dtype),
                ]
            )
            enc = {"input_ids": input_ids.unsqueeze(0)}

            out = _mdl_gpt(**enc, use_cache=False, return_dict=True)
            logits_pred = out.logits[0]
            tokens = _tok_gpt.convert_ids_to_tokens(enc["input_ids"][0].tolist())
            # we have to decode for display anyways, better to just do it here
            display_tokens = [detok_single(t) for t in tokens]

            topk_all = []
            actual = []
            logprob_sum = 0.0
            N = len(input_ids) - 1
            for i in range(N):
                probs = torch.softmax(logits_pred[i], dim=-1)
                topk_prob, topk_idx = torch.topk(probs, k=TOPK)
                topk = []
                for p, idx in zip(topk_prob.tolist(), topk_idx.tolist()):
                    topk.append(
                        {
                            "token": detok_single(
                                _tok_gpt.convert_ids_to_tokens([idx])[0]
                            ),
                            "prob": float(p),
                        }
                    )
                topk_all.append(topk)

                tid = int(input_ids[i + 1].item())
                p = float(probs[tid].item())
                lp = float(torch.log(probs[tid]).item())
                logprob_sum += lp
                actual.append(
                    {"token": display_tokens[i + 1], "prob": p, "logprob": lp, "pos": i}
                )

            return {
                "tokens": display_tokens[1:-1],
                "topk": topk_all[:-1],
                "actual": actual[:-1],
                "summary": {
                    "log_prob": logprob_sum,
                    "log10_prob": logprob_sum / math.log(10),
                    "avg_log_prob": logprob_sum / (N - 1),
                    "num_predicted": N - 1,
                },
            }


@app.post("/gpt2/next")
async def gpt2_next_endpoint(inp: Inp):
    async with QUEUE_LOCK:
        with torch.inference_mode():
            text, count, was_truncated = limit_chars((inp.text or ""))

            def detok_single(token_str: str) -> str:
                if token_str == "Ċ":
                    return "<new-line>"
                try:
                    return _tok_gpt.convert_tokens_to_string([token_str])
                except Exception:
                    return _tok_gpt.decode(_tok_gpt.convert_tokens_to_ids([token_str]))

            enc = _tok_gpt(text, return_tensors="pt", add_special_tokens=False)
            if enc["input_ids"].shape[1] == 0:
                input_ids = torch.tensor(
                    [_tok_gpt.bos_token_id or _tok_gpt.eos_token_id], dtype=int
                )
            else:
                # concat BOS + input_ids since GPT2 doesn't do that
                input_ids = enc["input_ids"][0]
                input_ids = torch.cat(
                    [
                        # EOS = BOS for GPT2 for some reason
                        # https://github.com/huggingface/transformers/issues/3311
                        torch.tensor(
                            [_tok_gpt.bos_token_id or _tok_gpt.eos_token_id],
                            dtype=input_ids.dtype,
                        ),
                        input_ids,
                    ]
                )
            enc = {"input_ids": input_ids.unsqueeze(0)}

            out = _mdl_gpt(**enc, use_cache=False, return_dict=True)
            logits = out.logits[0]
            probs = torch.softmax(logits[-1], dim=-1)
            topk_prob, topk_idx = torch.topk(probs, k=min(TOPK, probs.shape[-1]))

            topk = []
            for p, idx in zip(topk_prob.tolist(), topk_idx.tolist()):
                tok_str = _tok_gpt.convert_ids_to_tokens([idx])[0]
                topk.append(
                    {
                        "token": re.sub(" ", "\u2423", detok_single(tok_str)),
                        "id": int(idx),
                        "prob": float(p),
                    }
                )

            return {
                "input": {
                    "text": text,
                    "tokens": [
                        detok_single(t)
                        for t in _tok_gpt.convert_ids_to_tokens(input_ids.tolist())
                    ][1:],
                },
                "topk": topk,
                "note": ("Input truncated." if was_truncated else None),
            }


@app.post("/bert")
async def bert_endpoint(inp: Inp):
    async with QUEUE_LOCK:
        with torch.inference_mode():
            text, count, was_truncated = limit_chars(inp.text or "")
            print(f"{text=}")
            if count == 0:
                text = "The model should return attention."
            enc = _tok_bert(text, return_tensors="pt", add_special_tokens=False)
            print(f"{enc=}")
            out = _mdl_bert(**enc, output_attentions=True, return_dict=True)
            tokens = _tok_bert.convert_ids_to_tokens(enc["input_ids"][0].tolist())
            att_last = None
            if out.attentions and out.attentions[-1] is not None:
                last = out.attentions[-1][0]
                att_last = last.mean(dim=0).cpu().tolist()
            return {
                "tokens": tokens,
                "attn": att_last,
                "note": ("Input truncated." if was_truncated else None),
            }


app.mount("/", StaticFiles(directory=".", html=True), name="static")
