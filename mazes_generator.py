import os
import matplotlib.pyplot as plt
import networkx as nx
import random

def wilson_algorithm(G, start):
    U = {start}
    T = nx.Graph()
    while len(U) < len(G.nodes):
        u = random.choice(list(G.nodes - U))
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

def save_maze_image(T, width, height, file_path, line_width=8):
    fig, ax = plt.subplots(figsize=(width, height))
    pos = {(x, y): (y, -x) for x, y in T.nodes()}
    nx.draw(T, pos=pos, with_labels=False, node_size=0, ax=ax, width=line_width)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_and_save_mazes(output_folder, num_images, width=15, height=15, line_width=8):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i in range(0, num_images):
        T = generate_ust_maze(width, height)
        file_path = os.path.join(output_folder, f'maze_{i+1}.png')
        save_maze_image(T, width, height, file_path, line_width)
        if (i + 1) % 100 == 0:
            print(f'{i + 1} images generated...')


output_folder = 'mazes_data'
num_images = 60000
width, height = 15, 15
line_width = 8


generate_and_save_mazes(output_folder, num_images, width, height, line_width)

