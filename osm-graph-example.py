import pickle
import networkx as nx
from osm_graph import OsmGraph
import matplotlib.pyplot as plt

if __name__ == '__main__':
    og = OsmGraph(57.15, 65.53, dist=10000)
    with open('Tyumen.graph', 'wb') as f:
        pickle.dump(og, f)
    nd1 = og.find_node(57.1499942, 65.5424106)
    nd2 = og.find_node(57.1485774, 65.5400840)
    path = nx.dijkstra_path(og.graph, nd1, nd2)
    g = nx.subgraph(og.graph, path)
    ax = og.draw_graph()
    og.draw_graph(graph=g, ax=ax, edge_color='r', width=2)
    plt.show()
