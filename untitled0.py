import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

import networkx as nx
import matplotlib.pyplot as plt
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
    for i in range(width):
        for j in range(height):
            current_index = i * height + j
            if i < width - 1 and adj_matrix[current_index, current_index + height] == 1:
                G.add_edge((i, j), (i + 1, j))
            if j < height - 1 and adj_matrix[current_index, current_index + 1] == 1:
                G.add_edge((i, j), (i, j + 1))
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



def upper_flatten_to_adj_matrix(vector, num_nodes):
    batch_size = vector.size(0)
    adj_matrix = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.float32, device=vector.device)
    
    idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    
    for b in range(batch_size):
        adj_matrix[b, idx[0], idx[1]] = vector[b]
        adj_matrix[b, idx[1], idx[0]] = vector[b]
    
    return adj_matrix

def adj_matrix_to_upper_flatten(adj_matrix):
    batch_size, num_nodes, _ = adj_matrix.size()
    
    idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    
    upper_flatten = adj_matrix[:, idx[0], idx[1]]
    
    return upper_flatten



def upper_triangular_to_vector(adj_matrix, grid_shape):
    upper_triangular_vector = []
    width, height = grid_shape
    
    for i in range(width):
        for j in range(height):
            current_index = i * height + j
            if i < width - 1:
                upper_triangular_vector.append(adj_matrix[current_index, current_index + height])
            if j < height - 1:
                upper_triangular_vector.append(adj_matrix[current_index, current_index + 1])
    
    return upper_triangular_vector

def vector_to_upper_triangular(vector, grid_shape):
    width, height = grid_shape
    size = width * height
    adj_matrix = np.zeros((size, size), dtype=int)
    index = 0
    
    for i in range(width):
        for j in range(height):
            current_index = i * height + j
            if i < width - 1:
                adj_matrix[current_index, current_index + height] = vector[index]
                adj_matrix[current_index + height, current_index] = vector[index]
                index += 1
            if j < height - 1:
                adj_matrix[current_index, current_index + 1] = vector[index]
                adj_matrix[current_index + 1, current_index] = vector[index]
                index += 1
    
    return adj_matrix

def load_graphs(filename):
    with open(filename, 'rb') as file:
        graphs = pickle.load(file)
    return graphs

def graph_to_vector(G, grid_shape):
    width, height = grid_shape
    adj_matrix = nx.to_numpy_array(G)
    vector = upper_triangular_to_vector(adj_matrix, grid_shape)
    return vector

def vector_to_graph(vector, grid_shape):
    adj_matrix = vector_to_upper_triangular(vector, grid_shape)
    G = graph_from_adjacency_matrix(adj_matrix, grid_shape[0], grid_shape[1])
    return G

def visualize_graph_from_vector(vector, grid_shape, title="Graph from Vector"):
    G = vector_to_graph(vector, grid_shape)
    draw_maze_from_matrix(nx.to_numpy_array(G), grid_shape[0], grid_shape[1])

def discretenoise(train_adj_b_vec, sigma, device, grid_shape):
    train_adj_b_vec = train_adj_b_vec.to(device)
    batch_size, num_elements = train_adj_b_vec.size()

    grid_rows, grid_cols = grid_shape
    adjacency_mask = torch.zeros_like(train_adj_b_vec)
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            current_index = i * grid_cols + j

            if j < grid_cols - 1:
                right_index = current_index + 1
                adjacency_mask[:, current_index] = 1
                adjacency_mask[:, right_index] = 1

            if i < grid_rows - 1:
                bottom_index = current_index + grid_cols
                adjacency_mask[:, current_index] = 1
                adjacency_mask[:, bottom_index] = 1
    adj_mask = vector_to_upper_triangular(adjacency_mask, grid_shape)
    
    
    bernoulli_probs = sigma.unsqueeze(-1).expand(batch_size, num_elements)
    noise_vec = torch.bernoulli(bernoulli_probs) 
    noise_probs = torch.where(
        train_adj_b_vec > 1/2,
        1 - sigma.unsqueeze(-1),
        sigma.unsqueeze(-1)
    )
    
    
    train_adj_b_noisy_vec = torch.bernoulli(noise_probs) 
    train_adj_b_noise = vector_to_upper_triangular(train_adj_b_noisy_vec, grid_shape)
    train_adj_b_noise = train_adj_b_noise * adj_mask
    
    
    print(train_adj_b_noisy_vec.shape)
    train_adj_b_noisy_vec = adj_matrix_to_upper_flatten(train_adj_b_noise)
    print(train_adj_b_noisy_vec, 'train_adj_b_noise_vec')
    grad_log_noise_vec = torch.abs(-train_adj_b_vec + train_adj_b_noisy_vec)
    print(grad_log_noise_vec.shape)
    return train_adj_b_noisy_vec, grad_log_noise_vec

# Simulation
device = torch.device("cpu")
grid_shape = (3, 3)  # Exemple d'une grille 3x3
batch_size = 1
num_elements = grid_shape[0] * (grid_shape[1] - 1) + (grid_shape[0] - 1) * grid_shape[1]

# Vecteur d'adjacence binaire aléatoire pour la grille 3x3
train_adj_b_vec = torch.randint(0, 2, (batch_size, num_elements), dtype=torch.float32)
sigma = torch.tensor([0.5], dtype=torch.float32).to(device)

# Convertir le vecteur d'adjacence en matrice d'adjacence initiale
initial_adj_matrix = torch.tensor(adjacency_matrix(generate_ust_maze(3,3), 3, 3))



print(initial_adj_matrix.shape)
train_adj_b_vec = torch.tensor(upper_triangular_to_vector(initial_adj_matrix, grid_shape)).unsqueeze(0)

print(train_adj_b_vec.shape)

print("Graphe initial:")
draw_maze_from_matrix(initial_adj_matrix, grid_shape[0], grid_shape[1])

# Application de la fonction pour ajouter du bruit
train_adj_b_noisy_vec, grad_log_noise_vec = discretenoise(train_adj_b_vec, sigma, device, grid_shape)

# Convertir le vecteur d'adjacence bruyant en matrice d'adjacence
noisy_adj_matrix = vector_to_upper_triangular(train_adj_b_noisy_vec.numpy()[0], grid_shape)
print("Graphe après ajout de bruit:")
draw_maze_from_matrix(noisy_adj_matrix, grid_shape[0], grid_shape[1])
