"""Verify our cache builders feed the encoder exactly what upstream feeds it.

The failure this guards against is silent: a wrong input format still produces
finite features and trains fine, it just puts the encoder off-distribution.
That already happened once -- the first Llama cache was built on raw text while
the encoder had only ever seen chat-wrapped text.

Canonical upstream chain, for a document with no instruction:
    _convert_to_str("", text)     -> "!@#$%^&*()" + text
    prepare_for_tokenization(...) -> <chat user turn>{that}</chat user turn>
    tokenize(...)                 -> splits on the separator, embed_mask marks
                                     the trailing content tokens
    forward/get_pooling           -> mean over the embed_mask span
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.llm2vec_qwen3.supervised_encoder import (  # noqa: E402
    SEPARATOR,
    prepare_for_tokenization,
)
from tools.precompute_llm2vec_text_cache import prepare_text as llama_prepare  # noqa: E402

CAPTION = "a person walks forward and waves the right hand"
UPSTREAM_SEPARATOR = "!@#$%^&*()"


def upstream_convert_to_str(instruction: str, text: str) -> str:
    """Verbatim from llm2vec.LLM2Vec._convert_to_str."""
    return (
        f"{instruction.strip()} !@#$%^&*(){text}"
        if instruction
        else f"!@#$%^&*(){text}"
    )


def check_separator_contract() -> None:
    assert SEPARATOR == UPSTREAM_SEPARATOR, f"separator drift: {SEPARATOR!r}"
    ours = "" + SEPARATOR + CAPTION
    theirs = upstream_convert_to_str("", CAPTION)
    assert ours == theirs, f"\n ours  : {ours!r}\n theirs: {theirs!r}"
    print(f"[1] separator contract OK: {theirs[:24]!r}...")


def check_qwen3_format() -> None:
    text = prepare_for_tokenization(upstream_convert_to_str("", CAPTION))
    assert text.startswith("<|im_start|>user\n"), text[:40]
    assert text.endswith("<|im_end|>"), text[-20:]
    assert UPSTREAM_SEPARATOR in text, "separator was lost in the chat wrapper"
    assert CAPTION in text, "caption was lost in the chat wrapper"
    print(f"[2] Qwen3 ChatML format OK: {text!r}")


def check_llama_format() -> None:
    text = llama_prepare(CAPTION)
    assert text.startswith("<|start_header_id|>user<|end_header_id|>\n\n"), text[:50]
    assert text.endswith("<|eot_id|>"), text[-20:]
    assert CAPTION in text
    print(f"[3] Llama-3 chat format OK: {text!r}")


def check_empty_caption() -> None:
    """The CFG null row must still be a valid, non-degenerate input."""
    for name, fn in (("qwen3", lambda c: prepare_for_tokenization(upstream_convert_to_str("", c))),
                     ("llama3", llama_prepare)):
        text = fn("")
        assert text.strip(), f"{name} produced an empty string for the null caption"
        print(f"[4] {name} null caption OK: {text!r}")


def main() -> None:
    check_separator_contract()
    check_qwen3_format()
    check_llama_format()
    check_empty_caption()
    print("ALL_FORMAT_CHECKS_PASSED")


if __name__ == "__main__":
    main()
