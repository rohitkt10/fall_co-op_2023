import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class GraphVisualizer:
    """
    Graph Visualizer class to plot graphs and their properties
    """
    def __init__(self, graphs, graph_names, colors):
        """
        Parameters:
        -----------
        graphs: list
            List of graphs
        graph_names: list
            List of graph names
        colors: list
            List of colors to plot
        """
        self.graphs = graphs
        self.graph_names = graph_names
        self.colors = colors

    def plot_network_graph(self, ax, graph_index=0, with_labels=False, node_size=10):
        """
        Plot network graph

        Parameters:
        -----------
        graph_index: int
            Index of the graph to plot
        with_labels: bool
            Whether to plot labels or not
        node_size: int
            Size of the node
        """
        pos = nx.spring_layout(self.graphs[graph_index])
        nx.draw_networkx(self.graphs[graph_index], pos, with_labels=with_labels, node_size=node_size, ax=ax)
        ax.set_title(self.graph_names[graph_index])

    def plot_degree_distribution(self, ax):
        """
        Plot degree distribution
        """
        for g, l, c in zip(self.graphs, self.graph_names, self.colors):
            ax.plot(np.sort([j for _, j in g.degree()])[::-1], marker='.', alpha=0.7, label=l, color=c)
            ax.set_xlabel('Degree')
            ax.set_ylabel('Frequency')
            ax.set_title('Degree Distribution')
            ax.legend()

    def plot_clustering_coefficient_distribution(self, ax, show_avg=True):
        """
        Plot Clustering coefficient plot
        """
        for g, l, c in zip(self.graphs, self.graph_names, self.colors):
            clustering_coefficient = nx.average_clustering(g)
            # Create a histogram of clustering coefficients
            clustering_values = list(nx.clustering(g).values())
            ax.plot(sorted(clustering_values, reverse=True), marker='.', alpha=0.7, label=l, color=c)
            # ax.hist(clustering_values, bins=20, alpha=0.5, color=c, label=l)
            if show_avg:
                # Add a vertical line for the average clustering coefficient
                ax.axvline(x=clustering_coefficient, linestyle='--', color=c, 
                            label=f'avg clustering coeff ({clustering_coefficient:.2f})')
            ax.set_xlabel('Clustering Coefficient')
            ax.set_ylabel('Frequency')
            ax.set_title('Clustering Coefficient')
            ax.legend()

    def plot_betweenness_centrality_distribution(self, ax, show_avg=True):
        """
        Plot betweenness centrality plot 
        """
        for g, l, c in zip(self.graphs, self.graph_names, self.colors):
            betweenness_centrality = nx.betweenness_centrality(g)
            # Create a histogram of betweenness centrality values
            betweenness_values = list(betweenness_centrality.values())
            # ax.plot(sorted(betweenness_centrality, reverse=True), marker='.', alpha=0.7, label=l, color=c)
            ax.hist(betweenness_values, bins=20, alpha=0.5, color=c, label=l)
            if show_avg:
                # Add a vertical line for the average betweenness centrality
                avg_betweenness_centrality = np.mean(betweenness_values)
                ax.axvline(x=avg_betweenness_centrality, linestyle='--', color=c, 
                            label=f'avg betweenness centrality ({avg_betweenness_centrality:.3f})')
            ax.set_xlabel('Betweenness Centrality')
            ax.set_ylabel('Frequency')
            ax.set_title('Betweenness Centrality Distribution')
            ax.legend()

    def plot_modularity(self, ax):
        """
        Plot modularity
        """
        modularities = []
        for g in self.graphs:
            communities = nx.algorithms.community.greedy_modularity_communities(g)
            modularities.append(nx.algorithms.community.modularity(g, communities))

        ax.bar(self.graph_names, modularities, color=self.colors)
        ax.set_xlabel('Graph Type')
        ax.set_ylabel('Modularity')
        ax.set_title('Modularity')

    def plot_path_length_distribution(self, ax, graph_index=0):
        """
        Plot the distribution of shortest path lengths between nodes in the graph.
        Parameters:
        -----------
        graph_index: int
            Index of the graph to plot
        """
        path_lengths = []
        for node1 in self.graphs[graph_index].nodes():
            for node2 in self.graphs[graph_index].nodes():
                if node1 != node2:
                    path_lengths.append(nx.shortest_path_length(self.graphs[graph_index], node1, node2))
        path_length_counts = dict(zip(*np.unique(path_lengths, return_counts=True)))
        ax.bar(path_length_counts.keys(), path_length_counts.values())
        ax.set_xlabel('Shortest Path Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Path Length Distribution')
