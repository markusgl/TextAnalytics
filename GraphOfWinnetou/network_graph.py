import networkx as nx
import matplotlib.pyplot as plt
from networkx import nx_pydot

class NetworkGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_edge(self, node_1, node_2, weight):
        self.graph.add_edge(node_1, node_2, weight=weight)

    def draw_network(self):
        #nx.draw_networkx(self.graph, node_size=100, ode_color=range(len(self.graph)))
        #nx.draw_kamada_kawai(self.graph, node_size=400, with_labels=True, font_size=10, edge_color='b')
        #nx.draw_random(self.graph, with_labels=True)

        e_small = [(u, v) for (u, v, d) in self.graph.edges(data=True) if d['weight'] <= 3]
        e_middle = [(u, v) for (u, v, d) in self.graph.edges(data=True) if d['weight'] <= 6]
        e_large = [(u, v) for (u, v, d) in self.graph.edges(data=True) if d['weight'] > 6]
        #pos = nx.spring_layout(self.graph)  # positions for all nodes
        pos = nx.nx_pydot.graphviz_layout(self.graph, prog='dot')

        # nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=200)

        # edges
        nx.draw_networkx_edges(self.graph, pos, edgelist=e_small, width=1)
        nx.draw_networkx_edges(self.graph, pos, edgelist=e_middle, width=2)
        nx.draw_networkx_edges(self.graph, pos, edgelist=e_large, width=6)

        # labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_family='sans-serif')

        plt.axis('off')  # disable axis
        plt.show()

