"""Curated 100-seed list for the second HumanML3D cfg10 batch.

25 commonly used seeds + 25 random 3-digit + 25 random 4-digit + 25 random 5-digit,
none overlapping batch 1 (42-141). Deterministic (Random(20260819)).
"""
import random

used = set(range(42, 142))
# ordered by how commonly they appear in ML papers/codebases, not numerically
common = [0, 1, 2, 3, 7, 1234, 2020, 2021, 2022, 2023, 2024, 2025, 3407, 12345,
          31415, 54321, 666, 777, 888, 999, 1000, 1024, 2048, 314, 555,
          10, 13, 17, 21, 23]
common = [s for s in common if s not in used][:25]
rng = random.Random(20260819)
taken = set(common) | used


def pick(lo, hi, n):
    out = []
    while len(out) < n:
        v = rng.randint(lo, hi)
        if v not in taken:
            taken.add(v)
            out.append(v)
    return sorted(out)


d3, d4, d5 = pick(100, 999, 25), pick(1000, 9999, 25), pick(10000, 99999, 25)
seeds = common + d3 + d4 + d5
assert len(seeds) == 100 == len(set(seeds)) and not (set(seeds) & used)
with open("eval_results/seedlist_batch2_20260819.txt", "w") as f:
    f.write("\n".join(map(str, seeds)) + "\n")
print("常用25:", common)
print("3位25:", d3)
print("4位25:", d4)
print("5位25:", d5)
