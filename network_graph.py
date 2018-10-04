import networkx as nx
import matplotlib.pyplot as plt


class NetworkGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_edge(self, node_1, node_2):
        self.graph.add_edge(node_1, node_2)

    def draw_network(self):
        #nx.draw_networkx(self.graph, node_size=100, ode_color=range(len(self.graph)))
        nx.draw_kamada_kawai(self.graph, node_size=400, with_labels=True, font_size=10, edge_color='b')
        #nx.draw_random(self.graph, with_labels=True)
        plt.show()


if __name__ == '__main__':
    ng = NetworkGraph()
    ng.add_edge('A', 'B')
    ng.add_edge('A', 'C')
    ng.add_edge('B', 'A')
    ng.add_edge('B', 'D')
    ng.add_edge('B', 'C')
    ng.draw_network()


