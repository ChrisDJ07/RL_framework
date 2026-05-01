python graphml_manual/graphml_web_viewer.py \
  --graph data/la_trinidad_hazard_graph.graphml \
  --output graphml_manual/output/la_trinidad_hazard_graph.html

## New Nodes
3802985239
3802985237
3802999082
312973884
3802744706
6435189153
1926488400
3807851461

## Remove
1142052181
3803008212
1260719044
7890222929
7890222945
6377521024
1846909739
13320517147


## Add command
python graphml_manual/graphml_edit.py import-nodes \
  --graph subgraph_selection/candidates/output8_250n/tradeoff_subgraph_n242.graphml \
  --base-graph data/la_trinidad_hazard_graph.graphml \
  --node-id 3802985239 \
  --node-id 3802985237 \
  --node-id 3802999082 \
  --node-id 312973884 \
  --node-id 3802744706 \
  --node-id 6435189153 \
  --node-id 1926488400 \
  --node-id 3807851461 \
  --output graphml_manual/output/graph_with_imported_nodes.graphml

- Visualize
python graphml_manual/graphml_render.py \
  --graph graphml_manual/output/graph_with_imported_nodes.graphml \
  --output graphml_manual/output/tradeoff_subgraph_n250.png


## Merge 200 and 250 winner
1141864140
1141864243
1142052206
1142052245
3530013251
3530013261
3803008061
7912600758
7937630113
7994669419

python graphml_manual/graphml_edit.py import-nodes \
  --graph subgraph_selection/candidates/output8_250n/tradeoff_subgraph_n250.graphml \
  --base-graph data/la_trinidad_hazard_graph.graphml \
  --node-id 1141864140 \
  --node-id 1141864243 \
  --node-id 1142052206 \
  --node-id 1142052245 \
  --node-id 3530013251 \
  --node-id 3530013261 \
  --node-id 3803008061 \
  --node-id 7912600758 \
  --node-id 7937630113 \
  --node-id 7994669419 \
  --output graphml_manual/output/subgraph_n250_merge.graphml


python graphml_manual/graphml_render.py --graph "subgraph_selection/candidates/output11_merge_200&250/subgraph_n260_merge.graphml" --output graphml_manual/output/tradeoff_subgraph_n260.png