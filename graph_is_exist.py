import os
import pickle
import matplotlib.pyplot as plt
from osm_graph import OsmGraph

if __name__ == '__main__':
    if 'Tyumen.graph' in os.listdir():
        print('found')
        with open('Tyumen.graph', 'rb') as f:
            graph = pickle.load(f)
    else:
        print('loading')
        graph = OsmGraph(57.15, 65.53, dist=10000)
    graph.draw_graph()
    plt.show()
