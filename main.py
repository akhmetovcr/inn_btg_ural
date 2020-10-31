if __name__ == "__main__":
    import osmnx as ox
    import networkx as nx
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
    def filter_fat_edge(n1, n2, n3):
        return region_total_graph[n1][n2][n3].get('highway', '') == 'tertiary'

    # def filter_fat_node(n1):
    #     return True

    fat_graph = nx.MultiDiGraph(
        nx.subgraph_view(
            region_total_graph,
            filter_edge=filter_fat_edge,
            # filter_node=filter_fat_node
        )
    )
    fat_graph.remove_nodes_from(list(nx.isolates(fat_graph)))

    # fig2, ax2 = plt.subplots()
    ox.plot_graph(fat_graph)
    # mpl_edges_collection = nx.draw_networkx_edges(fat_graph, pos=nx.spring_layout(fat_graph))
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.add_collection(mpl_edges_collection)
    # plt.show()

    print("FINISH")
