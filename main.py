import osmnx as ox
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


def create_filtered_graph(graph, key, value):
    def filter_fat_edge(n1, n2, n3):
        return graph[n1][n2][n3].get(key, '') == value

    f_graph = nx.MultiDiGraph(
        nx.subgraph_view(
            region_total_graph,
            filter_edge=filter_fat_edge,
        )
    )
    f_graph.remove_nodes_from(list(nx.isolates(f_graph)))

    return f_graph


class RegularCluster:
    def __init__(self, ni, nj, north, east, south, west):
        # количество ячеек
        self.ni = ni
        self.nj = nj
        # границы кластера
        self.north = north
        self.east = east
        self.south = south
        self.west = west
        # значения в ячейках
        self.values = None

    def randomize(self, chance):
        self.values = np.fromfunction(
            np.vectorize(lambda i, j: int(random.random() < chance)),
            (self.ni, self.nj),
            dtype=np.int
        )

    def get_value(self, i, j):
        return self.values[i][j]


if __name__ == "__main__":
    # прямоугольник описанный вокруг Тюмени
    north, east, south, west = 57.2662, 65.9653, 57.0346, 65.2066
    # полный граф Тюмени
    region_total_graph = ox.graph_from_bbox(north=north, east=east, south=south, west=west, network_type='drive')
    # fig1, ax1 = plt.subplots()
    ox.plot_graph(region_total_graph)

    # # вариант не подходит, т.к. оказывается информации о ширине очень мало
    # def filter_fat_edge(n1, n2, n3):
    #     widths = region_total_graph[n1][n2][n3].get('width', '0')
    #     print(widths)
    #     if type(widths) is str:
    #         return int(widths) >= 10
    #     else:
    #         return False

    # берем только основные дороги (упрощено)
    fat_graph = create_filtered_graph(region_total_graph, 'highway', 'tertiary')

    # fig2, ax2 = plt.subplots()
    ox.plot_graph(fat_graph)
    # mpl_edges_collection = nx.draw_networkx_edges(fat_graph, pos=nx.spring_layout(fat_graph))
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.add_collection(mpl_edges_collection)
    # plt.show()

    # количество шагов по х и по у для кластеризации
    ni, nj = 50, 50
    # генерация значений 0 или 1 в кластерах
    cluster = RegularCluster(
        ni=ni, nj=nj,
        north=north, east=east, south=south, west=west
    )

    x = np.linspace(east, west, ni + 1)
    y = np.linspace(north, south, nj + 1)
    xg, yg = np.meshgrid(x, y)

    fig = plt.figure()
    for i in range(ni):
        for j in range(nj):
            if cluster.get_value(i, j):
                plt.fill([x[i], x[i + 1], x[i + 1], x[i], x[i]], [y[i], y[i], y[i + 1], y[i + 1], y[i]])

    plt.show()
    print("FINISH")
