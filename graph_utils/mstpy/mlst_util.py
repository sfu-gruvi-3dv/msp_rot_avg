from graph_utils.mstpy.graph import pylib_graph
from graph_utils.mstpy.edge_base import edge_base

def MLST_build_new_graph_with_black_edge(graph_t:pylib_graph, edges_in_black:list):
    graph_res = graph_t.getPorpertyCopy()
    for node in graph_t.node_list:
        graph_res.addNode(node)

    for eid in edges_in_black:
        edge = graph_t.edge_list[eid]
        graph_res.addEdge(edge)
    
    return graph_res
