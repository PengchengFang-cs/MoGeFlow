# Complex qualitative candidate pool, 2026-06-08

This folder contains skeleton preview pages for screening paper qualitative examples.

## Outputs

- `fig_candidates_complex_page01.{png,pdf}`: candidates C01-C04.
- `fig_candidates_complex_page02.{png,pdf}`: candidates C05-C08.
- `fig_candidates_complex_page03.{png,pdf}`: candidates C09-C12.

## Methods

- `Ours (PS-CF)`: `generation/qual_candidates_complex_20260608/ours_pscf_w200_bestfid_ema/results.json`
- `M-Transformer`: `generation/qual_candidates_complex_20260608/mtransformer/results.json`
- `MoMask`: `generation/qual_candidates_complex_20260608/momask/results.json`

## PS-CF sampling

- Checkpoint: `checkpoints/t2m/codeflow_part_structured_newvqtop3_w200_p256_h1536_d6s12_drop005_w0_b64_lr1e4_e600_eval10from0_seed42_test_h200_blossom_g0_20260606/model/best_fid.pt`
- Weight source: `ema`
- Steps: 96
- Cond scale: 6.0

## Candidate IDs

| ID | Prompt | Length |
| --- | --- | --- |
| C01 | a person walks forward, turns around, then walks back to the starting point. | 196 |
| C02 | a person runs forward, jumps once, then lands and continues jogging. | 160 |
| C03 | a person crouches down, picks something up from the floor, then stands up and turns left. | 196 |
| C04 | a person takes two steps forward, kicks with the right leg, then steps back. | 140 |
| C05 | a person walks in a circle while waving both arms above the head. | 180 |
| C06 | a person bends down, touches the ground with both hands, then stretches up. | 160 |
| C07 | a person sidesteps to the right, spins around, and sidesteps back to the left. | 180 |
| C08 | a person kneels on one knee, pauses, then stands up and raises both arms. | 196 |
| C09 | a person does jumping jacks, then turns around and walks away. | 196 |
| C10 | a person staggers forward as if dizzy, then regains balance and stands still. | 180 |
| C11 | a person performs a boxing combo with two punches and a front kick. | 160 |
| C12 | a person crawls forward on hands and knees, then gets up and walks forward. | 196 |

Initial shortlist for SMPL rendering: C03, C04, C05, C06, C08, C11.
Secondary options: C07, C10, C12.
