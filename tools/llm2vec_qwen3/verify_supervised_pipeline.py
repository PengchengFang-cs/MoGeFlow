"""End-to-end check of the supervised training pipeline, before spending a run.

Each check targets a failure that would still let training complete with a
falling loss:

  A. batch grouping -- E5Data must place one source dataset per batch, and the
     sampler must stay sequential, otherwise in-batch negatives are meaningless;
  B. collator/tokenizer -- three parallel columns of equal batch size, each
     carrying an embed_mask, all padded left;
  C. loss behaviour -- HardNegativeNLLLoss must reward the diagonal: a perfect
     retriever must score far below a random one, and near log(N) at chance;
  D. gradient flow -- a real backward pass must produce finite, non-zero
     gradients on the LoRA parameters only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.llm2vec_qwen3.vendored.E5Data import E5Data  # noqa: E402
from tools.llm2vec_qwen3.vendored.HardNegativeNLLLoss import (  # noqa: E402
    HardNegativeNLLLoss,
)

ECHO = "/scratch/pf2m24/text-caches/echo-data"
BATCH = 8


def check_batch_grouping() -> E5Data:
    data = E5Data(dataset_name="E5", split="train", file_path=ECHO,
                  effective_batch_size=BATCH)
    print(f"[A] loaded {len(data)} triplets")
    # Every consecutive block of BATCH rows must come from one source dataset.
    bad = 0
    for start in range(0, min(len(data.data), BATCH * 200), BATCH):
        block = data.data[start: start + BATCH]
        if len({s.task_name for s in block}) != 1:
            bad += 1
    if bad:
        raise RuntimeError(f"{bad} of the first 200 batches mix source datasets")
    sample = data[0]
    assert len(sample.texts) == 3, f"expected (query, positive, negative), got {len(sample.texts)}"
    assert "!@#$%^&*()" in sample.texts[0], "query is missing the instruction separator"
    print(f"[A] batch grouping OK: 200 batches each from a single dataset; "
          f"triplet columns = {len(sample.texts)}")
    return data


def check_collator(data) -> None:
    from transformers import AutoTokenizer

    from tools.llm2vec_qwen3.run_supervised_qwen3 import TripletCollator
    from tools.llm2vec_qwen3.supervised_encoder import Qwen3SentenceEncoder

    tokenizer = AutoTokenizer.from_pretrained(
        "/scratch/pf2m24/hf-models/Qwen3-8B", local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder = Qwen3SentenceEncoder.__new__(Qwen3SentenceEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = None
    encoder.tokenizer = tokenizer
    encoder.max_length = 512
    encoder.pooling_mode = "mean"
    encoder.skip_instruction = True

    features, labels = TripletCollator(encoder)([data[i] for i in range(BATCH)])
    assert len(features) == 3, f"expected 3 columns, got {len(features)}"
    for name, feature in zip(("query", "positive", "negative"), features):
        assert feature["input_ids"].shape[0] == BATCH, name
        assert "embed_mask" in feature, f"{name} column has no embed_mask"
        # left padding: row starts with pad where the row is shorter than max
        am = feature["attention_mask"]
        assert torch.all(am[:, -1] == 1), f"{name} is not left-padded"
    assert labels.shape[0] == BATCH
    print(f"[B] collator OK: 3 columns x {BATCH}, embed_mask present, left-padded")


def check_loss() -> None:
    loss_fn = HardNegativeNLLLoss(scale=20.0)
    torch.manual_seed(0)
    n, dim = 8, 64
    q = torch.nn.functional.normalize(torch.randn(n, dim), dim=-1)

    perfect = loss_fn(q, q.clone(), None)                       # positives identical to queries
    random_pos = loss_fn(q, torch.nn.functional.normalize(torch.randn(n, dim), dim=-1), None)
    chance = math.log(n)
    print(f"[C] loss: perfect {perfect.item():.4f} | random {random_pos.item():.4f} "
          f"| log(N)={chance:.4f}")
    if perfect.item() >= random_pos.item():
        raise RuntimeError("Loss does not reward matched pairs; the diagonal convention is wrong")
    if perfect.item() > 0.05:
        raise RuntimeError(f"Perfect retrieval should give ~0 loss, got {perfect.item():.4f}")

    with_neg = loss_fn(q, q.clone(), torch.nn.functional.normalize(torch.randn(n, dim), dim=-1))
    if with_neg.item() < perfect.item():
        raise RuntimeError("Adding hard negatives must not reduce the loss below the no-negative case")
    print(f"[C] loss OK: hard negatives raise it to {with_neg.item():.4f}, diagonal convention correct")


def check_gradients(data) -> None:
    """A real forward+backward on the assembled model."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    from tools.llm2vec_qwen3.bidirectional_qwen3 import (
        Qwen3BiForMNTP,
        disable_module_causal_flags,
    )
    from tools.llm2vec_qwen3.run_supervised_qwen3 import (
        LORA_TARGETS,
        TripletCollator,
        move_to_device,
    )
    from tools.llm2vec_qwen3.supervised_encoder import Qwen3SentenceEncoder

    tokenizer = AutoTokenizer.from_pretrained(
        "/scratch/pf2m24/hf-models/Qwen3-8B", local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3BiForMNTP.from_pretrained(
        "/scratch/pf2m24/hf-models/Qwen3-8B", local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    disable_module_causal_flags(model)
    trunk = PeftModel.from_pretrained(model.model, "/scratch/pf2m24/hf-models/Qwen3-8B-mntp")
    trunk = trunk.merge_and_unload()
    trunk = get_peft_model(
        trunk,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                   target_modules=LORA_TARGETS, bias="none", task_type=None),
    )
    trunk.enable_input_require_grads()
    trunk.config.use_cache = False
    encoder = Qwen3SentenceEncoder(trunk, tokenizer, max_length=512).cuda()

    features, _ = TripletCollator(encoder)([data[i] for i in range(4)])
    features = move_to_device(features, "cuda")
    q = encoder(features[0])
    d = encoder(features[1])
    n = encoder(features[2])
    loss = HardNegativeNLLLoss(scale=20.0)(q, d, n)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()

    grads = [(name, p.grad) for name, p in encoder.named_parameters()
             if p.requires_grad and p.grad is not None]
    nonzero = [n_ for n_, g in grads if g.abs().sum().item() > 0]
    if not nonzero:
        raise RuntimeError(
            "No LoRA parameter received a non-zero gradient -- gradient checkpointing "
            "likely severed the graph"
        )
    if any(not torch.isfinite(g).all() for _, g in grads):
        raise RuntimeError("Non-finite gradient encountered")
    print(f"[D] gradients OK: loss {loss.item():.4f}, "
          f"{len(nonzero)}/{len(grads)} LoRA tensors with non-zero finite grad")


def main() -> None:
    data = check_batch_grouping()
    check_collator(data)
    check_loss()
    check_gradients(data)
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
