if __name__ == "__main__":
    import osmnx as ox
    # прямоугольник описанный вокруг Тюмени
    north, east, south, west = 57.2662, 65.9653, 57.0346, 65.2066
    region_total_graph = ox.graph_from_bbox(north=north, east=east, south=south, west=west, network_type='drive')
    ox.plot_graph(region_total_graph)
