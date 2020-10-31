import osmnx as ox
import networkx as nx
import numpy as np


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


if __name__ == "__main__":

    # import matplotlib.pyplot as plt

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
    n_step_x, n_step_y = 50, 50
    # генерация значений 0 или 1 в кластерах
    map_values = np.fromfunction(
        np.vectorize(lambda i, j: int(i == j)),
        (n_step_x, n_step_y),
        dtype=np.int
    )
    x = np.linspace(east, west, n_step_x + 1)
    y = np.linspace(north, south, n_step_y + 1)
    xg, yg = np.meshgrid(x, y)

    fig = plt.figure()
    for i in range(n_step_x):
        for j in range(n_step_y):
            if map_values[i, j]:
                plt.fill([x[i], x[i + 1], x[i + 1], x[i], x[i]], [y[i], y[i], y[i + 1], y[i + 1], y[i]])

    plt.show()
    print("FINISH")
