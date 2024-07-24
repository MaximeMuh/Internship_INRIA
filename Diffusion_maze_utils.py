#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:02:34 2024

@author: mmuhleth
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def wilson_algorithm(G, start):
    U = {start}
    T = nx.Graph()
    while len(U) < len(G.nodes):
        u = random.choice(list(set(G.nodes) - U))
        path = [u]
        while u not in U:
            u = random.choice(list(G.neighbors(u)))
            if u in path:
                cycle_index = path.index(u)
                path = path[:cycle_index + 1]
            else:
                path.append(u)
        U.update(path)
        T.add_edges_from((path[i], path[i + 1]) for i in range(len(path) - 1))
    return T

def generate_ust_maze(width, height):
    G = nx.grid_2d_graph(width, height)
    start = (random.randint(0, width-1), random.randint(0, height-1))
    T = wilson_algorithm(G, start)
    return T

def draw_maze(T, width, height):
    pos = {(x, y): (x, y) for x, y in T.nodes()}
    plt.figure(figsize=(10, 10))
    nx.draw(T, pos=pos, with_labels=False, node_size=10, width=2, edge_color='blue')
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().invert_yaxis()
    plt.show()

def adjacency_matrix(T, width, height):
    nodes = [(i, j) for j in range(height) for i in range(width)]
    index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    adj_matrix = np.zeros((size, size), dtype=int)
    for edge in T.edges():
        i, j = index[edge[0]], index[edge[1]]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix

def graph_from_adjacency_matrix(adj_matrix, width, height):
    G = nx.Graph()
    size = len(adj_matrix)
    
    node_positions = [(i % width, i // width) for i in range(size)]
    G.add_nodes_from(node_positions)
    
    for i in range(size):
        for j in range(i + 1, size):
            if adj_matrix[i, j] == 1:
                node1 = node_positions[i]
                node2 = node_positions[j]
                G.add_edge(node1, node2)
    
    return G

def draw_maze_from_matrix(adj_matrix, width, height):
    G = graph_from_adjacency_matrix(adj_matrix, width, height)
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=pos, with_labels=False, node_size=10, width=2, edge_color='blue')
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().invert_yaxis()
    plt.show()


width, height = 10, 10  
T = generate_ust_maze(width, height)
draw_maze(T, width, height)

adj_matrix = adjacency_matrix(T, width, height)
print("Matrice d'adjacence:")
print(adj_matrix)


draw_maze_from_matrix(adj_matrix, width, height)


import numpy as np 
def upper_triangular_to_vector(adj_matrix):
    size = len(adj_matrix)
    upper_triangular_vector = []
    
    for i in range(size):
        for j in range(i + 1, size):  
            upper_triangular_vector.append(adj_matrix[i, j])
    
    return upper_triangular_vector

vector = upper_triangular_to_vector(adj_matrix)
print(vector)
def vector_to_upper_triangular(vector, size):
    adj_matrix = np.zeros((size, size), dtype=int)
    index = 0
    
    for i in range(size):
        for j in range(i + 1, size):
            adj_matrix[i, j] = vector[index]
            adj_matrix[j, i] = vector[index]  
            index += 1
    
    return adj_matrix



print(vector_to_upper_triangular(vector, width * height))

