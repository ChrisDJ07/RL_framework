# 260n Retraining Strategy

## What We Learned

### 1. Staging helps, especially at `200n`

Direct `200n` scratch existed:

- `results/runs/_no_folder/stage_control_1_200n.txt`

It eventually became good, but it was much slower than staged transfer:

- direct `200n` best eps=0 success: `92.0%` at episode `32200`
- staged `200n` best eps=0 success: `99.0%` at episode `17400`
- staged `200n_further` best eps=0 success: `100.0%` at episode `16800`

Sample-efficiency was even more decisive:

- direct `200n` first `90%` eps=0 success: episode `31600`
- staged `200n` first `90%` eps=0 success: episode `4500`
- staged `200n_further` first `90%` eps=0 success: episode `600`

### 2. Same-graph consolidation helps

This was true at both `150n` and `200n`.

- `stage_150` best eps=0 success: `89.5%`
- `stage_150_further` best eps=0 success: `99.5%`

- `stage_200` best eps=0 success: `99.0%`
- `stage_200_further` best eps=0 success: `100.0%`

### 3. Late branching works

The strongest profile warm runs branched from:

- `checkpoints/staged_training/stage_200_further/best_model.pt`

Examples:

- `results/runs/profile_training_new/last_run_stage_200_balanced_warm.txt`
- `results/runs/profile_training_new/last_run_stage_200_fast_warm.txt`
- `results/runs/profile_training_new/last_run_stage_200_safe_warm.txt`

These were much stronger than older profile branches.

### 4. Many runs overshot the peak

Several runs peaked well before the episode cap and then drifted or plateaued.
So the new strategy uses shorter episode caps, especially on transferred stages.

### 5. Old checkpoints are not safe to reuse blindly on the new graph family

This is the most important practical constraint.

The new chain:

- `data/final_subgraphs/subgraph_n100.graphml`
- `data/final_subgraphs/subgraph_n150.graphml`
- `data/final_subgraphs/subgraph_n200.graphml`
- `data/final_subgraphs/subgraph_n260.graphml`

is a different graph family from the old staged subgraphs.

The model contains a node embedding table whose rows depend on graph node indexing.
So old checkpoint paths are good **templates for hyperparameters**, but not safe **initialization sources** for the first new `100n` stage.

## Safe Changes We Can Make

### Safe change: clone configs as templates, not as checkpoint lineage

Best practice:

- copy the structure and hyperparameters from the strongest old configs
- reset the checkpoint chain to the new family
- start new `100n` from scratch
- transfer only within the new family from there onward

### Safe change: keep a shared backbone through `200_further`

Recommended backbone:

1. `100n`, `d=2`, no-HF, scratch
2. `150n`, `d=2`, no-HF, from new `100n`
3. `200n`, `d=3`, no-HF, from new `150n`
4. `200n_further`, `d=3`, no-HF, same graph

Then branch:

5. `260n Balanced`, `d=4`
6. `260n Fast`, `d=4`
7. `260n Safe`, `d=4`

### Safe change: move to hazard-independent speed

Set:

- `flood_time_weight = 0.0`
- `landslide_time_weight = 0.0`

This preserves:

- hazard-aware reward
- hazard-aware blocking
- RI-wide speed multiplier

but removes local hazard from travel-time distortion.

### Safe change: increase delivery count gradually

Recommended curriculum:

- `100n`: `2`
- `150n`: `2`
- `200n`: `3`
- `260n`: `4`

### Safe change: raise timeout budgets when deliveries increase

The new configs already do this.

## Episode Budget Recommendation

These caps are intentionally lower than many old runs because transferred stages often peaked early.

### Backbone

- `100n` scratch: `18000`
  - larger because this is the first stage in a new graph family
- `150n` transfer: `9000`
- `200n` transfer with `d=3`: `12000`
- `200n_further`: `8000`

### 260 warm profiles

- `Balanced`: `12000`
- `Fast`: `10000`
- `Safe`: `14000`

These are not guaranteed optimal, but they are a safer first-pass than inheriting old `15000-22000` caps everywhere.

## Fallback Logic

If the new `150 -> 200` jump is unstable:

- add `150_further` before `200`

Do **not** fall back first to direct `200n` scratch.
The history argues strongly against that as the main strategy.

## Files Added

Configs:

- `configs/retraining_260/backbone_stage_100_d2_nohf.json`
- `configs/retraining_260/backbone_stage_150_d2_nohf.json`
- `configs/retraining_260/backbone_stage_200_d3_nohf.json`
- `configs/retraining_260/backbone_stage_200_d3_nohf_further.json`
- `configs/retraining_260/warm_stage_260_balanced_d4_nohf.json`
- `configs/retraining_260/warm_stage_260_fast_d4_nohf.json`
- `configs/retraining_260/warm_stage_260_safe_d4_nohf.json`

Reference summary:

- `training_analysis/training_run_summary.csv`
- `training_analysis/retraining_strategy_260n.md`
