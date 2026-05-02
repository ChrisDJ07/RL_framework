## First
python subgraph_selection/select_tradeoff_subgraph.py --graphml data/la_trinidad_hazard_graph.graphml --num-nodes 200 --num-pairs 10000 --candidate-seeds 40 --rain-keys RI1

## Second
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 200 \
  --num-pairs 20000 \
  --candidate-seeds 60 \
  --rain-keys RI1

## Third
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 200 \
  --num-pairs 50000 \
  --candidate-seeds 100 \
  --rain-keys RI1


## Fourth RI23
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 200 \
  --num-pairs 20000 \
  --candidate-seeds 60 \
  --rain-keys RI2,RI3

## Fifth RI23 
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 200 \
  --num-pairs 50000 \
  --candidate-seeds 100 \
  --rain-keys RI2,RI3

## Sixth
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 200 \
  --num-pairs 75000 \
  --candidate-seeds 250 \
  --rain-keys RI1

## 300n
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 300 \
  --num-pairs 50000 \
  --candidate-seeds 200 \
  --rain-keys RI1

## 250n
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 250 \
  --num-pairs 75000 \
  --candidate-seeds 250 \
  --rain-keys RI1

## Ninth
python subgraph_selection/select_tradeoff_subgraph.py \
  --graphml data/la_trinidad_hazard_graph.graphml \
  --num-nodes 200 \
  --num-pairs 90000 \
  --candidate-seeds 350 \
  --rain-keys RI1