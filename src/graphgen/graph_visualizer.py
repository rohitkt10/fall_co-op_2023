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

    def plot_clustering_coefficient_distribution(self, ax):
        """
        Plot Clustering coefficient plot
        """
        for g, l, c in zip(self.graphs, self.graph_names, self.colors):
            clustering_coefficient = nx.average_clustering(g)
            # Create a histogram of clustering coefficients
            clustering_values = list(nx.clustering(g).values())
            ax.plot(sorted(clustering_values, reverse=True), marker='.', alpha=0.7, label=l, color=c)
            # ax.hist(clustering_values, bins=20, alpha=0.5, color=c, label=l)
            # Add a vertical line for the average clustering coefficient
            ax.axvline(x=clustering_coefficient, linestyle='--', color=c, 
                        label=f'avg clustering coeff ({clustering_coefficient:.2f})')
            ax.set_xlabel('Clustering Coefficient')
            ax.set_ylabel('Frequency')
            ax.set_title('Clustering Coefficient')
            ax.legend()

    def plot_betweenness_centrality_distribution(self, ax):
        """
        Plot betweenness centrality plot 
        """
        for g, l, c in zip(self.graphs, self.graph_names, self.colors):
            betweenness_centrality = nx.betweenness_centrality(g)
            # Create a histogram of betweenness centrality values
            betweenness_values = list(betweenness_centrality.values())
            ax.hist(betweenness_values, bins=20, alpha=0.5, color=c, label=l)
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