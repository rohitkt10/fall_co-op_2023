
import sys
sys.path.append(".")

import networkx as nx
import logging
import matplotlib.pyplot as plt
from src.graphgen.graph_with_motifs_dataset import *

logging.basicConfig(filename='graphGenDataset_log.log', level=logging.INFO, force=True)


graph = GraphWithMotifsDataset(
    num_graphs=1000,
    min_nodes=30,
    max_nodes=40,
    base_graph_model='BA',
    motif_graph_model='WS',
    p=0.1,
    m=2,
    k=3,
    min_motif_size=3,
    max_motif_size=4,
    min_num_motifs=1,
    max_num_motifs=1,
)
graph.generate_dataset()
dataset = graph.get_dataset()


for label in set([label for _, _, _, label in dataset]):
    graphs = [(nx.Graph(g), m, f) for g, m, f, l in dataset if l == label]
    fig, axs = plt.subplots(1, len(graphs), figsize=(20, 4))
    fig.suptitle(f"Graphs with label {label}")
    for i, (g, m, f) in enumerate(graphs):
        # print(f'Graph {i+1}, nodes: {len(g.nodes())}, edges: {len(g.edges())}, node features: {f}, label: {label}')
        # get non-zero entry in node features
        node_labels = {k:np.argmax(v) for k, v in f.items()}
        axs[i].set_title(f"Graph {i+1}")
        pos = nx.spring_layout(g)
        # there is only one motif in the list for this binary classification problem
        node_colors = ['lightcoral' if node in m[0] else 'skyblue' for node in g.nodes]
        nx.draw_networkx(g, pos=pos, node_color=node_colors, ax=axs[i], labels=node_labels)
    plt.show()