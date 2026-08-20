"""Print the paired raw/whitened geometry diagnostic in LaTeX row form."""
import json

d = json.load(open("eval_results/diag_20260809/whitened_rho_test.json"))
NAMES = {"root_lower_spine": "root", "upper_torso_arms": "upper arms", "right_leg": "right leg",
         "upper_spine_neck": "upper neck", "left_leg": "left leg", "head": "head"}
widths = []
for r in d["rows"]:
    m = r["metrics"]
    raw, wht, shuf = m["raw"]["spearman"], m["whitened"]["spearman"], m["raw_shuffled"]["spearman"]
    widths.append(m["raw"]["spearman_ci_high"] - m["raw"]["spearman_ci_low"])
    print("{:10s} & {:3d} & {:.3f} & {:.3f} & {:+.3f} \\\\".format(
        NAMES[r["part_name"]], r["active_codes"], raw, wht, shuf))
mo = d["mean_over_parts"]["feature"]
print("Mean & {:.1f} & {:.3f} & {:.3f} & {:+.3f} \\\\".format(
    sum(r["active_codes"] for r in d["rows"]) / 6,
    mo["spearman_raw_mean"], mo["spearman_whitened_mean"], mo["spearman_raw_shuffled_mean"]))
print("max CI width = {:.3f}".format(max(widths)))
