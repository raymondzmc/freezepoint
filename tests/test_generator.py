"""Tests for the generators, the prompt templates and the reply parser. The API generator runs against a stub client;
the live API test and the local-model test are deselected by default."""
import sys
from types import SimpleNamespace

import pytest

from harness.factors import ALLOWED_OPS
from loop import generator
from loop.generator import (
    DEFAULT_FAKE_EXPRS,
    ApiGenerator,
    FakeGenerator,
    LocalGenerator,
    make_generator,
    parse_expression,
    render,
)

PROPOSE = {"operators": "Mean", "fields": "$close", "context": "csi300, 2018-01-01 to 2021-12-31"}
REFINE = {"operators": "Mean", "fields": "$close", "expr": "$close", "feedback": "ic 0.01", "history": "none"}

# parse_expression
def test_parse_expression_takes_last_fenced_block():
    assert parse_expression("reasoning\n```\nMean($close, 5)/$close - 1\n```\n") == "Mean($close, 5)/$close - 1"
def test_parse_expression_several_blocks_takes_the_last():
    assert parse_expression("```\n$open\n```\nbetter:\n```\n$close\n```") == "$close"
def test_parse_expression_language_tagged_fence():
    assert parse_expression("here\n```python\nMean($close, 5)\n```") == "Mean($close, 5)"
def test_parse_expression_unfenced_trailing_line():
    assert parse_expression("I think this works.\nRank($close)\nthat is all") == "Rank($close)"
    assert parse_expression("text\n($close - $open) / $open") == "($close - $open) / $open"
    assert parse_expression("text\n$close / $open\n") == "$close / $open"
def test_parse_expression_nothing_found_returns_stripped_text():
    assert parse_expression("  I cannot do that.  \n") == "I cannot do that."
def test_parse_expression_empty_reply():
    assert parse_expression("") == "" and parse_expression("   \n") == ""

# render
def test_prompt_templates_render_without_missing_keys():
    p, r = render("propose", PROPOSE), render("refine", REFINE)
    assert "csi300" in p and "ic 0.01" in r
    for op in ALLOWED_OPS: assert op in render("propose", {**PROPOSE, "operators": ", ".join(sorted(ALLOWED_OPS))})
def test_render_raises_on_missing_key():
    with pytest.raises(KeyError): render("propose", {"operators": "Mean", "fields": "$close"})
    with pytest.raises(KeyError): render("refine", {k: v for k, v in REFINE.items() if k != "history"})

# FakeGenerator
def test_fake_generator_cycles_and_is_seed_stable():
    g = FakeGenerator(["$close", "$open"]); assert g.propose("", 3, 0) == ["$close", "$open", "$close"]
    assert FakeGenerator(["$close", "$open"]).propose("", 3, 0) == ["$close", "$open", "$close"]
def test_fake_generator_seeds_start_in_different_places_and_cost_is_empty():
    g = FakeGenerator(["a", "b", "c"])
    assert FakeGenerator(["a", "b", "c"]).propose("", 1, 1) == ["b"] and g.propose("", 1, 5) == ["c"] and g.last_cost == {}
def test_fake_generator_cursor_is_shared_by_propose_and_refine():
    g = FakeGenerator(["a", "b", "c"])
    assert g.propose("", 2, 0) == ["a", "b"]
    assert g.refine("a", {}, [], 0) == "c" and g.refine("c", {}, [], 0) == "a"   # the seed only matters on the first call

# ApiGenerator
def reply(text="```\n$close\n```", stop_reason="end_turn", blocks=None, category=None):
    """A Messages API response as the SDK returns it: content blocks, stop reason and details, model, usage."""
    content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
    details = SimpleNamespace(type="refusal", category=category, explanation="declined") if stop_reason == "refusal" \
        else None
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=details,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))

class StubClient:
    """Each reply is a string (an `end_turn` text reply) or a response built by `reply`."""
    def __init__(self, replies, model="model-returned"):
        self.calls, self._replies, self._model = [], list(replies), model
        self.messages = self
    def create(self, **kw):
        self.calls.append(kw)
        r = self._replies.pop(0)
        r = reply(r) if isinstance(r, str) else r
        r.model = self._model
        return r

def api(replies, effort=None, **kw):
    stub = StubClient(replies); return ApiGenerator("model-x", None, 777, effort=effort, client=stub, **kw), stub

def test_api_request_has_exactly_the_intended_keys_and_the_rendered_prompt():
    g, stub = api(["```\n$close\n```"])
    assert g.propose("window ctx", 1, 3) == ["$close"]
    (kw,) = stub.calls
    assert set(kw) == {"model", "max_tokens", "messages"}          # no temperature, no seed, no effort, nothing else
    assert kw["model"] == "model-x" and kw["max_tokens"] == 777
    assert kw["messages"] == [{"role": "user", "content": render("propose", generator.propose_values("window ctx"))}]
    assert g.last_cost["effort"] == "api-default"
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_api_effort_is_sent_in_output_config_and_recorded(effort):
    g, stub = api(["```\n$close\n```"], effort=effort)
    assert g.refine("$close", {}, [], 0) == "$close"
    (kw,) = stub.calls
    assert set(kw) == {"model", "max_tokens", "messages", "output_config"} and kw["output_config"] == {"effort": effort}
    assert g.last_cost["effort"] == effort
@pytest.mark.parametrize("effort", ["", "HIGH", "minimal", 3, True])
def test_api_refuses_an_effort_the_api_does_not_document(effort):
    with pytest.raises(ValueError, match="effort"): ApiGenerator("model-x", None, 100, effort=effort, client=StubClient([]))
def test_api_propose_makes_n_requests_and_sums_usage():
    g, stub = api(["```\n$close\n```", "```\n$open\n```", "no expression here"])
    assert g.propose("ctx", 3, 0) == ["$close", "$open", "no expression here"] and len(stub.calls) == 3
    assert g.last_cost["input_tokens"] == 30 and g.last_cost["output_tokens"] == 15
    assert g.last_cost["model"] == "model-returned" and g.last_cost["sampling"] == "api-default"
    assert g.last_cost["seconds"] >= 0 and g.last_cost["stop_reasons"] == ["end_turn"] * 3
def test_api_reply_with_no_text_block_is_the_empty_candidate_with_its_stop_reason():
    """An `end_turn` reply that holds only a thinking block: the model wrote nothing, which the validator rejects as an
    empty expression; the ledger can tell it apart from a cut-off or a refusal by the recorded stop reason."""
    g, _ = api([reply(blocks=[SimpleNamespace(type="thinking", thinking="")])])
    assert g.propose("ctx", 1, 0) == [""] and g.last_cost["stop_reasons"] == ["end_turn"]
@pytest.mark.parametrize("stop, words", [("refusal", ["refusal", "cyber"]), ("max_tokens", ["max_tokens", "777"]),
                                         ("model_context_window_exceeded", ["model_context_window_exceeded"]),
                                         ("pause_turn", ["pause_turn"])])
def test_api_reply_that_did_not_end_its_turn_raises_and_is_no_candidate(stop, words):
    g, stub = api(["```\n$open\n```", reply("```\n$cl", stop_reason=stop, category="cyber")])
    g.last_cost = {"before": True}
    with pytest.raises(generator.GenerationError) as e:
        g.propose("ctx", 2, 0)
    assert all(w in str(e.value) for w in words) and len(stub.calls) == 2
    assert g.last_cost == {"before": True}                         # nothing recorded for a call that produced nothing
    with pytest.raises(generator.GenerationError, match=stop):
        api([reply(stop_reason=stop)])[0].refine("$close", {}, [], 0)
def test_api_refine_prompt_shows_feedback_and_last_three_history_entries():
    g, stub = api(["```\nRank($close)\n```"])
    hist = [{"expr": f"e{i}", "ic": 0.01 * i} for i in range(4)] + [{"expr": "bad(", "error": "unknown operator bad"}]
    assert g.refine("$close", {"ic": 0.02, "icir": 0.5, "ls_sharpe": 1.1, "turnover": 0.3}, hist, 0) == "Rank($close)"
    text = stub.calls[0]["messages"][0]["content"]
    assert "e0" not in text and "e1" not in text and "e2" in text and "e3" in text
    assert "unknown operator bad" in text and "0.5" in text and "1.1" in text and "0.3" in text
    assert g.last_cost["input_tokens"] == 10
def test_api_refuses_a_temperature():
    with pytest.raises(ValueError, match="generator.temperature must be null"):
        ApiGenerator("model-x", 0.7, 100, client=StubClient([]))
def test_api_missing_key_raises_at_call_time_not_construction(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(generator, "ENV_FILE", tmp_path / ".env")
    g = ApiGenerator("model-x", None, 100)                          # building one needs no key
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"): g.propose("ctx", 1, 0)
def test_api_key_falls_back_to_dotenv_and_env_wins(monkeypatch, tmp_path):
    env = tmp_path / ".env"; env.write_text("# c\nOTHER=1\nANTHROPIC_API_KEY='from-file'\n")
    monkeypatch.setattr(generator, "ENV_FILE", env)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert generator.read_api_key() == "from-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env"); assert generator.read_api_key() == "from-env"
def test_api_client_is_built_with_three_sdk_retries(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert ApiGenerator("model-x", None, 100)._get_client().max_retries == 3

# LocalGenerator
def test_local_generator_names_the_extra_when_libraries_are_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)                 # forces ImportError even if installed
    with pytest.raises(ImportError, match="uv sync --extra models"): LocalGenerator("pit-4b-ft-201912")

class FakeTensor:
    """The slice of a torch tensor the generator touches: `.shape` and `[row, slice]`."""
    def __init__(self, rows): self.rows = rows
    @property
    def shape(self): return (len(self.rows), len(self.rows[0]))
    def __getitem__(self, key): row, sl = key; return self.rows[row][sl]

class FakeEncoding(dict):
    """Like transformers 5's BatchEncoding (a mapping with `.to(device)`), which `apply_chat_template` returns."""
    device = None
    def to(self, device): self.device = device; return self

def test_local_generator_passes_the_encoding_decodes_new_tokens_only_and_is_greedy(monkeypatch):
    log = {"seeds": [], "generate": [], "decoded": [], "templates": [], "model_kw": None}
    reply = "think\n```\nRank($close)\n```"
    class Tok:
        @classmethod
        def from_pretrained(cls, path): log["tok_path"] = path; return cls()
        def apply_chat_template(self, messages, **kw):
            log["templates"].append((messages, kw))
            return FakeEncoding(input_ids=FakeTensor([[1, 2, 3]]), attention_mask=FakeTensor([[1, 1, 1]]))
        def decode(self, ids, skip_special_tokens=False): log["decoded"].append(list(ids)); return reply
    class Model:
        device = "dev"
        @classmethod
        def from_pretrained(cls, path, **kw): log["model_kw"] = kw; return cls()
        def generate(self, *args, **kw):
            log["generate"].append((args, kw)); return FakeTensor([[1, 2, 3, 7, 8]])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=log["seeds"].append))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tok, AutoModelForCausalLM=Model))
    g = LocalGenerator("pit-test", max_new_tokens=64)
    assert g.propose("ctx", 2, 5) == ["Rank($close)", "Rank($close)"]
    assert log["seeds"] == [5, 5] and log["tok_path"] == generator.MODELS / "pit-test"
    assert log["model_kw"]["dtype"] == "auto" and "torch_dtype" not in log["model_kw"]
    messages, kw = log["templates"][0]
    assert kw["return_dict"] is True and kw["return_tensors"] == "pt" and kw["add_generation_prompt"] is True
    assert messages == [{"role": "user", "content": render("propose", generator.propose_values("ctx"))}]
    args, gkw = log["generate"][0]
    assert args == () and set(gkw) == {"input_ids", "attention_mask", "max_new_tokens", "do_sample"}
    assert gkw["do_sample"] is False and gkw["max_new_tokens"] == 64
    assert log["decoded"][0] == [7, 8]                                  # only the newly generated tokens
    assert g.last_cost["input_tokens"] == 3 and g.last_cost["output_tokens"] == 2 and g.last_cost["sampling"] == "greedy"
    assert g.refine("$close", {"ic": 0.1}, [], 9) == "Rank($close)" and log["seeds"][-1] == 9

# make_generator and the default fake expressions
def gen_cfg(provider, **kw):
    from loop.config import Generator
    return Generator(**{"provider": provider, "model": "m-1", "revision": "m-1-rev", "temperature": None,
                        "max_output_tokens": 321, **kw})
def test_make_generator_api_needs_no_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(generator, "ENV_FILE", tmp_path / ".env")
    g = make_generator(gen_cfg("api"))
    assert isinstance(g, ApiGenerator) and g.model == "m-1" and g.max_output_tokens == 321 and g._client is None
    assert g.effort is None and make_generator(gen_cfg("api", effort="high")).effort == "high"
def test_make_generator_api_refuses_a_temperature():
    with pytest.raises(ValueError, match="temperature"): make_generator(gen_cfg("api", temperature=0.5))
def test_make_generator_local_passes_the_model_name(monkeypatch):
    seen = []
    monkeypatch.setattr(generator, "LocalGenerator", lambda name: seen.append(name) or "local-generator")
    assert make_generator(gen_cfg("local")) == "local-generator" and seen == ["m-1"]
def test_make_generator_fake_returns_the_configured_exprs():
    g = make_generator(gen_cfg("fake", exprs=("$close", "$open")))
    assert isinstance(g, FakeGenerator) and g.propose("", 3, 0) == ["$close", "$open", "$close"]
def test_make_generator_unknown_provider():
    with pytest.raises(ValueError, match="provider"): make_generator(gen_cfg("openai"))
def test_default_fake_exprs_all_valid_but_one_that_reads_the_future(monkeypatch, tmp_path):
    from harness import factors
    from harness.pit import tools
    monkeypatch.setattr(tools, "QLIB_DATA", tmp_path / "no-qlib-data")        # no dry run: data-free
    verdicts = {e: factors.validate(e, "cn") for e in DEFAULT_FAKE_EXPRS}
    bad = [e for e, (ok, _) in verdicts.items() if not ok]
    assert len(DEFAULT_FAKE_EXPRS) >= 6 and len(set(DEFAULT_FAKE_EXPRS)) == len(DEFAULT_FAKE_EXPRS)
    assert len(bad) == 1 and "future" in verdicts[bad[0]][1]
    assert DEFAULT_FAKE_EXPRS.index(bad[0]) == 1                     # seed 0 meets it on its second generation

@pytest.mark.api
def test_api_live_one_proposal():
    from loop import config
    tokens = config.load(generator.ROOT / "configs" / "cn_refine.yaml").generator.max_output_tokens
    out = ApiGenerator("claude-haiku-5-5", None, tokens).propose("csi300, daily, 2018-2021", 1, 0)
    assert len(out) == 1 and out[0]

@pytest.mark.gpu
def test_local_generator_is_greedy_and_seed_stable():
    g = LocalGenerator("pit-4b-ft-201912")
    assert g.propose("csi300, daily", 1, 0) == g.propose("csi300, daily", 1, 0)
