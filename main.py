import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from osmnx.plot import utils_graph
import random

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


def plot_graph(
    G,
    ax=None,
    figsize=(8, 8),
    bgcolor="#111111",
    node_color="w",
    node_size=15,
    node_alpha=None,
    node_edgecolor="none",
    node_zorder=1,
    edge_color="#999999",
    edge_linewidth=1,
    edge_alpha=None
):
    """
    Plot a graph.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    ax : matplotlib axis
        if not None, plot on this preexisting axis
    figsize : tuple
        if ax is None, create new figure with size (width, height)
    bgcolor : string
        background color of plot
    node_color : string or list
        color(s) of the nodes
    node_size : int
        size of the nodes: if 0, then skip plotting the nodes
    node_alpha : float
        opacity of the nodes, note: if you passed RGBA values to node_color,
        set node_alpha=None to use the alpha channel in node_color
    node_edgecolor : string
        color of the nodes' markers' borders
    node_zorder : int
        zorder to plot nodes: edges are always 1, so set node_zorder=0 to plot
        nodes below edges
    edge_color : string or list
        color(s) of the edges' lines
    edge_linewidth : float
        width of the edges' lines: if 0, then skip plotting the edges
    edge_alpha : float
        opacity of the edges, note: if you passed RGBA values to edge_color,
        set edge_alpha=None to use the alpha channel in edge_color
    show : bool
        if True, call pyplot.show() to show the figure
    close : bool
        if True, call pyplot.close() to close the figure
    save : bool
        if True, save the figure to disk at filepath
    filepath : string
        if save is True, the path to the file. file format determined from
        extension. if None, use settings.imgs_folder/image.png
    dpi : int
        if save is True, the resolution of saved file
    bbox : tuple
        bounding box as (north, south, east, west). if None, will calculate
        from spatial extents of plotted geometries.

    Returns
    -------
    fig, ax : tuple
        matplotlib figure, axis
    """
    max_node_size = max(node_size) if hasattr(node_size, "__iter__") else node_size
    max_edge_lw = max(edge_linewidth) if hasattr(edge_linewidth, "__iter__") else edge_linewidth
    if max_node_size <= 0 and max_edge_lw <= 0:
        raise ValueError("Either node_size or edge_linewidth must be > 0 to plot something.")

    # create fig, ax as needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bgcolor, frameon=False)
        ax.set_facecolor(bgcolor)
    else:
        fig = ax.figure

    if max_edge_lw > 0:
        # plot the edges' geometries
        gdf_edges = utils_graph.graph_to_gdfs(G, nodes=False)["geometry"]
        ax = gdf_edges.plot(ax=ax, color=edge_color, lw=edge_linewidth, alpha=edge_alpha, zorder=1)

    if max_node_size > 0:
        # scatter plot the nodes' x/y coordinates
        gdf_nodes = utils_graph.graph_to_gdfs(G, edges=False, node_geometry=False)[["x", "y"]]
        ax.scatter(
            x=gdf_nodes["x"],
            y=gdf_nodes["y"],
            s=node_size,
            c=node_color,
            alpha=node_alpha,
            edgecolor=node_edgecolor,
            zorder=node_zorder,
        )

    return fig, ax


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
    _, ax = plot_graph(fat_graph)
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
    cluster.randomize(0.1)

    x = np.linspace(east, west, ni + 1)
    y = np.linspace(north, south, nj + 1)
    xg, yg = np.meshgrid(x, y)

    for i in range(ni):
        for j in range(nj):
            if cluster.get_value(i, j):
                ax.fill([x[i], x[i + 1], x[i + 1], x[i], x[i]], [y[i], y[i], y[i + 1], y[i + 1], y[i]])

    plt.show()
    print("FINISH")
