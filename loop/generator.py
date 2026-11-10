"""Generators: the one place a language model is asked for a factor expression.

Three implementations share one interface: `FakeGenerator` (tests), `ApiGenerator` (Anthropic Messages API) and
`LocalGenerator` (a time-locked model under `data/models/`). Guarantees: `propose` and `refine` return exactly what
`parse_expression` extracts from the model's reply, or the stripped reply itself when nothing can be extracted (never
text the model did not write, never an exception), so the caller can record it and the validator can reject it. The
operator and field lists in the prompts come from `harness.factors`. The API request carries no sampling parameter and no
seed (the Messages API accepts neither), so the config must say so: a non-null temperature is refused at construction.
The API key is read only at the first call, never at import or construction."""
from __future__ import annotations

import os
import pathlib
import re
import time
from typing import Protocol

from harness.factors import ALLOWED_FIELDS, ALLOWED_OPS

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = pathlib.Path(__file__).parent / "prompts"
ENV_FILE = ROOT / ".env"
MODELS = ROOT / "data" / "models"
HISTORY_SHOWN = 3
FEEDBACK_KEYS = ("ic", "icir", "ls_sharpe", "turnover")

_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_EXPR_LINE = re.compile(r"^(?:\$|\(|(?:" + "|".join(sorted(ALLOWED_OPS)) + r")\b)")

class Generator(Protocol):
    last_cost: dict
    def propose(self, context: str, n: int, seed: int) -> list[str]: ...
    def refine(self, expr: str, feedback: dict, history: list[dict], seed: int) -> str: ...

def parse_expression(text: str) -> str:
    """The last non-empty fenced block, else the last line starting with `$`, `(` or an operator name, else the
    stripped text itself. Never raises."""
    text = text or ""
    for block in reversed(_FENCE.findall(text)):
        if block.strip(): return block.strip()
    for line in reversed(text.splitlines()):
        if _EXPR_LINE.match(line.strip()): return line.strip()
    return text.strip()

def render(name: str, values: dict) -> str:
    """Fill `loop/prompts/<name>.md`; a missing key raises KeyError rather than rendering a blank."""
    return (PROMPTS / f"{name}.md").read_text().format_map(values)

def _lists() -> dict:
    return {"operators": ", ".join(sorted(ALLOWED_OPS)), "fields": ", ".join(sorted(ALLOWED_FIELDS))}

def propose_values(context: str) -> dict:
    return {**_lists(), "context": context}

def _num(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

def format_history(history: list[dict]) -> str:
    """The last three entries; each has `expr` and either `ic` or an `error` string (an invalid attempt)."""
    shown = history[-HISTORY_SHOWN:]
    if not shown: return "none yet"
    return "\n".join(f"{i}. {h['expr']}  ->  " + (f"invalid: {h['error']}" if h.get("error") else f"ic {_num(h.get('ic'))}")
                     for i, h in enumerate(shown, 1))

def refine_values(expr: str, feedback: dict, history: list[dict]) -> dict:
    fb = "\n".join(f"{k}: {_num(feedback.get(k))}" for k in FEEDBACK_KEYS)
    return {**_lists(), "expr": expr, "feedback": fb, "history": format_history(history)}

class FakeGenerator:
    """Cycles through `exprs`. The cursor starts at `seed % len(exprs)` on the first call and advances by one for
    every expression returned, by `propose` and `refine` alike."""
    def __init__(self, exprs: list[str]):
        self.exprs, self.last_cost, self._cursor = list(exprs), {}, None
    def _next(self, seed: int) -> str:
        if self._cursor is None: self._cursor = seed % len(self.exprs)
        out = self.exprs[self._cursor % len(self.exprs)]; self._cursor += 1
        return out
    def propose(self, context: str, n: int, seed: int) -> list[str]:
        self.last_cost = {}; return [self._next(seed) for _ in range(n)]
    def refine(self, expr: str, feedback: dict, history: list[dict], seed: int) -> str:
        self.last_cost = {}; return self._next(seed)

def read_api_key() -> str:
    """`ANTHROPIC_API_KEY` from the environment, else from `.env` at the repo root (plain KEY=VALUE lines)."""
    if key := os.environ.get("ANTHROPIC_API_KEY"): return key
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == "ANTHROPIC_API_KEY" and v.strip().strip("'\""): return v.strip().strip("'\"")
    raise RuntimeError(f"ANTHROPIC_API_KEY is not set: export it or put ANTHROPIC_API_KEY=... in {ENV_FILE}")

class ApiGenerator:
    """The Anthropic Messages API. One request per expression. No sampling parameter and no seed is sent: the API
    accepts neither, so `temperature` must be None and `last_cost["sampling"]` records "api-default". Retries are the
    SDK's own (3, covering 408/409/429/5xx and connection errors)."""
    def __init__(self, model: str, temperature: float | None, max_output_tokens: int, client=None):
        if temperature is not None:
            raise ValueError("the Messages API no longer accepts sampling parameters, so generator.temperature must be "
                             f"null for the api provider (got {temperature!r}); a config that says otherwise would "
                             "misdescribe the run")
        self.model, self.max_output_tokens, self.last_cost, self._client = model, max_output_tokens, {}, client
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=read_api_key(), max_retries=3)
        return self._client
    def _ask(self, prompts: list[str]) -> list[str]:
        client, t0, tokens_in, tokens_out, served = self._get_client(), time.monotonic(), 0, 0, None
        out = []
        for prompt in prompts:
            r = client.messages.create(model=self.model, max_tokens=self.max_output_tokens,
                                       messages=[{"role": "user", "content": prompt}])
            tokens_in += r.usage.input_tokens; tokens_out += r.usage.output_tokens; served = r.model
            out.append(parse_expression("".join(b.text for b in r.content if b.type == "text")))
        self.last_cost = {"input_tokens": tokens_in, "output_tokens": tokens_out, "seconds": time.monotonic() - t0,
                          "model": served, "sampling": "api-default"}
        return out
    def propose(self, context: str, n: int, seed: int) -> list[str]:
        return self._ask([render("propose", propose_values(context))] * n)
    def refine(self, expr: str, feedback: dict, history: list[dict], seed: int) -> str:
        return self._ask([render("refine", refine_values(expr, feedback, history))])[0]

class LocalGenerator:
    """A local causal LM from `data/models/<name>`, greedy decoding with the seed set before each generation."""
    def __init__(self, name: str, max_new_tokens: int = 1024):
        try:
            import torch
            import transformers
        except ImportError as e:
            raise ImportError("LocalGenerator needs torch and transformers: run `uv sync --extra models`") from e
        self._torch, self.name, self.max_new_tokens, self.last_cost = torch, name, max_new_tokens, {}
        path = MODELS / name
        self._tok = transformers.AutoTokenizer.from_pretrained(path)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="auto")
    def _generate(self, prompt: str, seed: int) -> str:
        t0 = time.monotonic(); self._torch.manual_seed(seed)
        # transformers 5: `apply_chat_template` returns a BatchEncoding (`return_dict` defaults to True); asked for explicitly.
        enc = self._tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(self._model.device)
        n_prompt = enc["input_ids"].shape[-1]
        out = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.last_cost = {"input_tokens": int(n_prompt), "output_tokens": int(out.shape[-1] - n_prompt),
                          "seconds": time.monotonic() - t0, "model": self.name, "sampling": "greedy"}
        return parse_expression(self._tok.decode(out[0, n_prompt:], skip_special_tokens=True))
    def propose(self, context: str, n: int, seed: int) -> list[str]:
        return [self._generate(render("propose", propose_values(context)), seed) for _ in range(n)]
    def refine(self, expr: str, feedback: dict, history: list[dict], seed: int) -> str:
        return self._generate(render("refine", refine_values(expr, feedback, history)), seed)
