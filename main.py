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
            graph,
            filter_edge=filter_fat_edge,
        )
    )
    f_graph.remove_nodes_from(list(nx.isolates(f_graph)))

    return f_graph


def create_filtered_graph_by_cluster(graph, cluster):
    def filter_intersect_edge(n1, n2, n3):
        if 'geometry' not in graph[n1][n2][n3]:
            return False
            print()
        xy = graph[n1][n2][n3]['geometry'].xy
        x1 = xy[0][0]
        y1 = xy[1][0]
        x2 = xy[0][-1]
        y2 = xy[1][-1]
        return cluster.is_intersect_with_any_non_zero(x1=x1, y1=y1, x2=x2, y2=y2)

    f_graph = nx.MultiDiGraph(
        nx.subgraph_view(
            graph,
            filter_edge=filter_intersect_edge,
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
        # шаг
        self.i_step = (east - west) / ni
        self.j_step = (south - north) / nj
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

    def is_intersect(self, i, j, x1, y1, x2, y2):
        x_left = self.west + i * self.i_step
        x_right = self.west + (i + 1) * self.i_step
        y_top = self.north + j * self.j_step
        y_bottom = self.north + (j + 1) * self.j_step

        # проверка лежит ли первая точка в прямоугольнике
        if x_left <= x1 <= x_right and y_bottom <= y1 <= y_top:
            return True

        # проверка лежит ли вторая точка в прямоугольнике
        if x_left <= x2 <= x_right and y_bottom <= y2 <= y_top:
            return True

        # # коэффициенты уравнения прямой входного сегмента
        # A_segment = y2 - y1
        # B_segment = x1 - x2
        # D_segment = x2 * y1 - x1 * y2
        # # коэффициенты уравнения прямой первой диагонали
        # A_diag1 = y_bottom - y_top
        # B_diag1 = x_left - x_right
        # D_diag1 = x_right * y_top - x_left * y_bottom
        # # коэффициенты уравнения прямой второй сегмента
        # A_diag2 = y_top - y_bottom
        # B_diag2 = x_left - x_right
        # D_diag2 = x_right * y_bottom - x_left * y_top
        # # если уравнение диагонали пересекается с уравнением отрезка
        # # и точки попарно лежат по разную сторону, то пересечение с диагональю есть
        # if (A_segment * x_left + B_segment * y_top + D_segment) * (A_segment * x_right + B_segment * y_bottom + D_segment) < 0 and (A_diag1 * x1 + B_diag1 * y1 + D_diag1) * (A_diag1 * x2 + B_diag1 * y2 + D_diag1) < 0:
        #     return True
        # if (A_segment * x_left + B_segment * y_bottom + D_segment) * (A_segment * x_right + B_segment * y_top + D_segment) < 0 and (A_diag2 * x1 + B_diag2 * y1 + D_diag1) * (A_diag2 * x2 + B_diag2 * y2 + D_diag2) < 0:
        #     return True

        return False

    def is_intersect_with_any_non_zero(self, x1, y1, x2, y2):
        for (i, j), value in np.ndenumerate(self.values):
            if value != 0 and self.is_intersect(i=i, j=j, x1=x1, x2=x2, y1=y1, y2=y2):
                return True
        return False


def main():
    # прямоугольник описанный вокруг Тюмени
    north, east, south, west = 57.2662, 65.9653, 57.0346, 65.2066
    # полный граф Тюмени
    region_total_graph = ox.graph_from_bbox(north=north, east=east, south=south, west=west, network_type='drive')

    # берем только основные дороги (упрощено)
    fat_graph = create_filtered_graph(region_total_graph, 'highway', 'tertiary')

    # кластеризуем карту
    # количество шагов по х и по у для кластеризации
    ni, nj = 10, 10
    # генерация значений 0 или 1 в кластерах
    cluster = RegularCluster(
        ni=ni, nj=nj,
        north=north, east=east, south=south, west=west
    )
    cluster.randomize(0.1)

    # фильтруем граф по ненулевым ячейкам кластера
    intersect_graph = create_filtered_graph_by_cluster(graph=fat_graph, cluster=cluster)

    bgcolor = "#111111"
    bbox = (north, south, east, west)
    fig, axes = plt.subplots(2, 2, facecolor=bgcolor, frameon=False)
    axes[0][0].set_facecolor(bgcolor)
    axes[0][1].set_facecolor(bgcolor)
    axes[1][0].set_facecolor(bgcolor)
    axes[1][1].set_facecolor(bgcolor)
    ox.plot_graph(region_total_graph, axes[0][0], show=False, bbox=bbox)
    ox.plot_graph(fat_graph, axes[0][1], show=False, bbox=bbox)
    ox.plot_graph(intersect_graph, axes[1][0], show=False, bbox=bbox)

    x = np.linspace(west, east, ni + 1)
    y = np.linspace(north, south, nj + 1)
    yg, xg = np.meshgrid(y, x)

    for i in range(ni):
        for j in range(nj):
            if cluster.get_value(i, j):
                axes[0][1].fill(
                    [xg[i, j], xg[i + 1, j], xg[i + 1, j + 1], xg[i, j + 1], xg[i, j]],
                    [yg[i, j], yg[i + 1, j], yg[i + 1, j + 1], yg[i, j + 1], yg[i, j]],
                    '#0000FF77'
                )
                axes[1][0].fill(
                    [xg[i, j], xg[i + 1, j], xg[i + 1, j + 1], xg[i, j + 1], xg[i, j]],
                    [yg[i, j], yg[i + 1, j], yg[i + 1, j + 1], yg[i, j + 1], yg[i, j]],
                    '#0000FF77'
                )

    plt.show()


if __name__ == "__main__":
    main()
    print("FINISH")
