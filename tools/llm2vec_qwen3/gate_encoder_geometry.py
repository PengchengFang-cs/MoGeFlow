#!/usr/bin/env python3
"""Gate: is a trained Qwen3-LLM2Vec encoder actually usable, before caching it.

Supervised contrastive training reports a falling loss whether or not the
resulting space is useful, and the cache builder's own checks only prove the
cache faithfully records whatever the encoder produced.  So this runs the one
measurement that separates the two: motion captions that mean the same thing
must land next to each other, and closer than captions that mean something
else.

Method: encode paraphrase pairs, subtract the mean direction (motion captions
share a large common component -- without removing it every pair looks similar
and the check cannot fail), then require that every sentence's nearest
neighbour is its own paraphrase.  Strict retrieval must be N/N: with 12 pairs,
chance is 1/23 per sentence, so a partial pass is a real failure, not noise.

The v1 encoder scored margin +0.6981 with 12/12 after supervised training, and
+0.2712 with 4/12 after MNTP alone -- so the threshold discriminates the two
stages it is meant to discriminate.

Usage:
    python tools/llm2vec_qwen3/gate_encoder_geometry.py \
        --base_model ... --mntp_model ... --supervised_model ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.llm2vec_qwen3.bidirectional_qwen3 import (  # noqa: E402
    Qwen3BiForMNTP,
    disable_module_causal_flags,
)
from tools.llm2vec_qwen3.supervised_encoder import (  # noqa: E402
    SEPARATOR,
    Qwen3SentenceEncoder,
    prepare_for_tokenization,
)

# Paraphrase pairs in the vocabulary of the datasets we condition on.
PAIRS = [
    ("a person walks forward.", "someone moves ahead on foot."),
    ("a man jumps up high.", "a person leaps upward."),
    ("the person sits down on a chair.", "someone takes a seat."),
    ("a person waves with their right hand.", "someone greets with the right arm."),
    ("a man runs quickly in a straight line.", "a person sprints straight ahead."),
    ("the person kicks with the left leg.", "someone strikes out using the left foot."),
    ("a person turns around to the left.", "someone rotates counter-clockwise."),
    ("a man picks something up from the floor.", "a person lifts an object off the ground."),
    ("the person climbs up a ladder.", "someone ascends a ladder."),
    ("a person throws a ball forward.", "someone tosses a ball ahead."),
    ("a man crouches down low.", "a person squats close to the ground."),
    ("the person dances in place.", "someone performs a dance without moving away."),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--mntp_model", required=True)
    p.add_argument("--supervised_model", default="")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--min_margin", type=float, default=0.40,
                   help="Required gap between paraphrase and non-paraphrase similarity.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3BiForMNTP.from_pretrained(
        args.base_model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    disable_module_causal_flags(model)
    trunk = model.model
    for path in [args.mntp_model] + ([args.supervised_model] if args.supervised_model else []):
        trunk = PeftModel.from_pretrained(trunk, path)
        trunk = trunk.merge_and_unload()
        print(f"[setup] merged {Path(path).name}", flush=True)
    encoder = Qwen3SentenceEncoder(trunk, tokenizer, max_length=int(args.max_length)).cuda().eval()

    sentences = [s for pair in PAIRS for s in pair]
    vecs = []
    with torch.no_grad():
        for s in sentences:  # batch=1: left padding shifts RoPE and changes vectors
            batch = {k: v.cuda() for k, v in
                     encoder.tokenize([prepare_for_tokenization("" + SEPARATOR + s)]).items()}
            vecs.append(encoder(batch)[0].float())
    x = torch.stack(vecs)
    x = x - x.mean(dim=0, keepdim=True)      # remove the shared caption direction
    x = F.normalize(x, dim=-1)
    sim = x @ x.T
    n = len(sentences)
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)

    partner = torch.arange(n, device=sim.device) ^ 1   # 0<->1, 2<->3, ...
    pos = sim[torch.arange(n), partner]
    neg_mask = ~eye.clone()
    neg_mask[torch.arange(n), partner] = False
    neg = sim.masked_select(neg_mask).view(n, n - 2).mean(dim=1)
    margin = float((pos - neg).mean())

    ranked = sim.masked_fill(eye, -2.0).argmax(dim=1)
    hits = int((ranked == partner).sum())

    print(f"GEOMETRY pairs={len(PAIRS)} margin={margin:+.4f} strict_retrieval={hits}/{n}")
    for i in range(0, n, 2):
        ok = "ok " if ranked[i] == partner[i] and ranked[i + 1] == partner[i + 1] else "MISS"
        print(f"  {ok} pos={float(pos[i]):+.3f}  {sentences[i][:46]!r}")

    if hits != n:
        raise SystemExit(f"GEOMETRY_GATE_FAILED strict retrieval {hits}/{n}, need {n}/{n}")
    if margin < args.min_margin:
        raise SystemExit(f"GEOMETRY_GATE_FAILED margin {margin:+.4f} < {args.min_margin}")
    print("GEOMETRY_GATE_OK")


if __name__ == "__main__":
    main()
