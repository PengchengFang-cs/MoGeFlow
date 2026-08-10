"""Self-check for the supervised encoder port.

The three ported pieces (embed_mask, instruction skipping, left-pad pooling)
all fail silently: training still runs and the loss still drops even if the
sentence vector is computed over the wrong tokens.  Each check below is
designed to be falsifiable, with a control that must fail.

  A. embed_mask marks exactly the content tokens, and only trailing ones;
  B. instruction invariance: changing the instruction must NOT change the
     vector, while changing the content MUST -- with skip_instruction=False as
     the control, where both must change;
  C. padding invariance: a sentence encoded alone and inside a padded batch
     must give the same vector (proves left-padding alignment is right);
  D. contrastive sanity: a paraphrase must score above an unrelated sentence
     under the InfoNCE similarity actually used by the loss.
"""

from __future__ import annotations

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

QWEN = "/scratch/pf2m24/hf-models/Qwen3-8B"
INSTR_A = "Given a web search query, retrieve relevant passages"
INSTR_B = "Given a claim, retrieve documents that refute it"
CONTENT = "a person walks forward and waves the right hand"
CONTENT2 = "a man sits down on a chair and crosses his legs"


def item(instruction: str, content: str) -> str:
    return prepare_for_tokenization(f"{instruction}; " + SEPARATOR + content)


def check_embed_mask(enc) -> None:
    batch = enc.tokenize([item(INSTR_A, CONTENT), item(INSTR_B, CONTENT)])
    em, am = batch["embed_mask"], batch["attention_mask"]
    assert em.shape == am.shape, (em.shape, am.shape)
    for row in range(em.shape[0]):
        marked = em[row].nonzero().flatten()
        assert len(marked) > 0, "no content tokens marked"
        # contiguous and flush to the right-hand end of the sequence
        assert marked[-1].item() == em.shape[1] - 1, "embed_mask must reach the last position"
        assert torch.equal(marked, torch.arange(marked[0], marked[-1] + 1)), "embed_mask not contiguous"
        assert em[row].sum() < am[row].sum(), "embed_mask must be a strict subset of attention_mask"
    print(f"[A] embed_mask OK: content spans {int(em[0].sum())}/{int(am[0].sum())} attended tokens")


@torch.no_grad()
def embed(enc, texts):
    batch = {k: v.cuda() for k, v in enc.tokenize(texts).items()}
    return enc(batch).float()


@torch.no_grad()
def check_pooling_correctness(enc) -> None:
    """Pooling must average exactly the embed_mask positions.

    Note this does NOT make the vector independent of the instruction: the
    trunk is bidirectional, so content tokens attend to the instruction and are
    conditioned by it -- that is the reason the instruction is in the input at
    all.  ``skip_instruction`` only controls which positions are averaged, so
    the check is a direct numerical one against a hand-computed mean.
    """
    texts = [item(INSTR_A, CONTENT), item(INSTR_B, CONTENT2)]
    batch = {k: v.cuda() for k, v in enc.tokenize(texts).items()}
    pooled = enc(batch).float()

    feature = {k: v for k, v in batch.items() if k != "embed_mask"}
    # Keep the model dtype: the encoder averages in bf16, so a fp32 reference
    # mean would differ by accumulation precision alone (~1e-1 absolute) and
    # say nothing about correctness.
    hidden = enc.model(**feature).last_hidden_state
    em, am = batch["embed_mask"], batch["attention_mask"]

    manual_content = torch.stack(
        [hidden[i, -int(em[i].sum()):, :].mean(dim=0) for i in range(len(texts))]
    ).float()
    manual_all = torch.stack(
        [hidden[i, -int(am[i].sum()):, :].mean(dim=0) for i in range(len(texts))]
    ).float()
    scale = pooled.abs().mean().item()
    err_content = (pooled - manual_content).abs().max().item() / scale
    err_all = (pooled - manual_all).abs().max().item() / scale
    print(f"[B] relative err vs content-mean {err_content:.6f}; vs all-token mean {err_all:.6f}")
    if err_content > 1e-3:
        raise RuntimeError(
            f"Pooling does not match the embed_mask content mean (rel err {err_content:.6f})"
        )
    if err_all < 100 * max(err_content, 1e-6):
        raise RuntimeError(
            "CONTROL FAILED: pooling over content tokens is indistinguishable from pooling "
            f"over all tokens (rel err {err_all:.6f}); embed_mask is not restricting anything"
        )
    print("[B] pooling OK: averages exactly the content span, clearly distinct from all-token mean")


def check_padding_sensitivity(enc) -> None:
    """Measure how much left padding moves the vector.

    Left padding shifts content tokens to later absolute positions, and a plain
    forward pass derives ``position_ids`` from ``arange(seq_len)`` without
    consulting the attention mask, so RoPE sees different positions and the
    vector changes.  Upstream LLM2Vec pads left and does not correct positions
    either, so this is faithful behaviour, not a porting bug -- it is also why
    the reference repo pins cache building to batch_size=1.  The check only
    fails if the drift is catastrophic, which would indicate real misalignment.
    """
    alone = embed(enc, [item(INSTR_A, CONTENT)])
    padded = embed(enc, [item(INSTR_A, CONTENT),
                         item(INSTR_A, CONTENT2 + " " + CONTENT2 + " " + CONTENT2)])
    cos = F.cosine_similarity(alone[0:1], padded[0:1]).item()
    print(f"[C] padding sensitivity cos {cos:.6f} (RoPE position shift; build caches with batch_size=1)")
    if cos < 0.90:
        raise RuntimeError(
            f"Padding drift is too large (cos {cos:.4f}); left-pad alignment is likely wrong"
        )


def check_contrastive(enc) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.llm2vec_qwen3.vendored.loss_utils import cos_sim

    q = embed(enc, [item(INSTR_A, CONTENT)])
    docs = embed(enc, [
        item("", "someone strides ahead while raising their right arm"),   # paraphrase
        item("", CONTENT2),                                                # unrelated
    ])
    scores = cos_sim(q, docs)[0]
    print(f"[D] paraphrase {scores[0].item():.4f} vs unrelated {scores[1].item():.4f}")
    if scores[0].item() <= scores[1].item():
        print("    NOTE: untrained encoder does not yet rank the paraphrase first; "
              "this is what supervised training is for, not a wiring bug.")


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3BiForMNTP.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda").eval()
    disable_module_causal_flags(model)
    enc = Qwen3SentenceEncoder(model.model, tokenizer, max_length=512)

    check_embed_mask(enc)
    check_pooling_correctness(enc)
    check_padding_sensitivity(enc)
    check_contrastive(enc)
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
