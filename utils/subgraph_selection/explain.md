We’ve got the first version built.

## What’s now in `subgraph_selection/`

Added:

- `subgraph_selection/__init__.py`
- `subgraph_selection/graph_io.py`
- `subgraph_selection/tradeoff.py`
- `subgraph_selection/selection.py`
- `subgraph_selection/select_tradeoff_subgraph.py`

## What it does

This pipeline now:

1. loads the full La Trinidad graph
2. samples OD pairs from the full graph
3. computes for each pair:
   - fastest path
   - safest path
   - relative time penalty of safety
   - relative hazard savings of safety
   - path divergence
4. scores each pair by tradeoff richness
5. aggregates those high-tradeoff paths into corridor weights
6. grows connected candidate subgraphs around strong corridor seeds
7. scores those candidates
8. exports the best connected subgraph for a requested node budget

So yes, this is built around the idea we discussed:
- not cherry-picking endpoint nodes
- but using fastest-vs-safest path structure to find a better connected training graph

## The hazard model it uses for selection

It uses the more safety-focused structure we talked about:

- convex hazard transform:
  - flood `0.0, 0.2, 0.6, 1.0 -> 0, 1, 4, 9`
  - landslide `0.0, 0.3, 0.6, 1.0 -> 0, 1, 4, 9`
- RI severity multipliers:
  - `RI1 1.00`
  - `RI2 1.25`
  - `RI3 1.60`
  - `RI4 2.10`
  - `RI5 2.80`

That is only for **subgraph scoring/selection**, which is the right place to use it.

## The main entrypoint

Use:

```bash
python subgraph_selection/select_tradeoff_subgraph.py --graphml data/la_trinidad_hazard_graph.graphml --num-nodes 200

python subgraph_selection/select_tradeoff_subgraph.py --graphml data/la_trinidad_hazard_graph.graphml --num-nodes 100 --num-pairs 1000 --candidate-seeds 10 --rain-keys RI1 --dry-run

python subgraph_selection/select_tradeoff_subgraph.py --graphml data/la_trinidad_hazard_graph.graphml --num-nodes 200 --num-pairs 10000 --candidate-seeds 40 --rain-keys RI1 --dry-run


```

## Most important options

- `--num-nodes`
  - required
  - this is the flexible node budget you asked for
  - examples: `100`, `150`, `200`, later `250`, etc.

- `--num-pairs`
  - how many OD pairs to sample on the full graph
  - default: `5000`

- `--rain-keys`
  - default: `RI2,RI3,RI4`
  - these are the severities used to evaluate tradeoff richness

- `--candidate-seeds`
  - how many strong corridor regions to try growing from
  - default: `30`

- `--min-time-minutes`
  - filters out trivial short OD pairs
  - default: `5.0`

- `--max-time-minutes`
  - optional upper bound for sampled pair difficulty
  - `0` disables it

- `--output-dir`
  - default: `subgraph_selection/output`

- `--dry-run`
  - useful to inspect rankings before writing files

## Example runs

### 100-node candidate
```powershell
.\.venv\Scripts\python.exe subgraph_selection\select_tradeoff_subgraph.py --graphml data\la_trinidad_hazard_graph.graphml --num-nodes 100 --num-pairs 6000 --rain-keys RI2,RI3,RI4
```

### 150-node candidate
```powershell
.\.venv\Scripts\python.exe subgraph_selection\select_tradeoff_subgraph.py --graphml data\la_trinidad_hazard_graph.graphml --num-nodes 150 --num-pairs 8000 --rain-keys RI2,RI3,RI4
```

### 200-node candidate
```powershell
.\.venv\Scripts\python.exe subgraph_selection\select_tradeoff_subgraph.py --graphml data\la_trinidad_hazard_graph.graphml --num-nodes 200 --num-pairs 10000 --rain-keys RI2,RI3,RI4
```

### Quick inspection only
```powershell
.\.venv\Scripts\python.exe subgraph_selection\select_tradeoff_subgraph.py --graphml data\la_trinidad_hazard_graph.graphml --num-nodes 150 --num-pairs 1000 --candidate-seeds 10 --dry-run
```

## Outputs

For a run like `--num-nodes 200`, it writes into `subgraph_selection/output/`:

- `tradeoff_subgraph_n200.graphml`
- `tradeoff_subgraph_n200_summary.json`
- `tradeoff_subgraph_n200.png`

The summary JSON includes:
- parameters used
- best candidate scores
- top candidate list
- top scored OD pairs

## Validation done

I already checked:

- syntax via `py_compile`
- a pair-scoring smoke test
- a full dry run on the La Trinidad graph

The dry run completed successfully.

## One important note

In the tiny dry-run sample, some candidate subgraphs only had a very small number of internal scored pairs.
That is not a bug, it’s just because:
- `--num-pairs` was intentionally small for validation

For real selection runs, we should use a much larger pair sample:
- at least `5000`
- probably `8000–15000` for the 1447-node graph

That will make the candidate scoring much more stable.

## My recommendation for the first serious run

Start with:

```powershell
.\.venv\Scripts\python.exe subgraph_selection\select_tradeoff_subgraph.py --graphml data\la_trinidad_hazard_graph.graphml --num-nodes 200 --num-pairs 10000 --candidate-seeds 40 --rain-keys RI2,RI3,RI4
```

Then inspect:
- the selected graph plot
- the JSON summary
- whether the chosen region looks geographically coherent

## What I’d do next

1. Run one serious `200`-node selection
2. Inspect the selected region visually
3. If it looks coherent, also generate `150` and maybe `250`
4. Then we can compare these new graphs structurally before deciding which one to train on

If you want, I can help with the next step immediately after you run the first serious selection and paste the summary back.