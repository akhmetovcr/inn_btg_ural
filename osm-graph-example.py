import networkx as nx
from osm_graph import OsmGraph
import matplotlib.pyplot as plt


if __name__ == '__main__':
    og = OsmGraph(35.7158, 139.8741)
    nd1 = og.find_node(35.7165, 139.8738)
    nd2 = og.find_node(35.7153, 139.8752)
    path = nx.dijkstra_path(og.graph, nd1, nd2)
    g = nx.subgraph(og.graph, path)
    ax = og.draw_graph()
    og.draw_graph(graph=g, ax=ax, edge_color='r', width=2)
    plt.show()
