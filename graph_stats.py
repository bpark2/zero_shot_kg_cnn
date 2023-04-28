import json
import math 

with open("data/graph_struct/graph_adj_dict.json") as graph_file: 
    adj_dict = json.load(graph_file)

# find the number of nodes
n_nodes = len(adj_dict.keys())
print(f"Number of nodes: {n_nodes}")

# find the number of edges, min deg, max deg, avg deg
min_deg = (math.inf, 0)
max_deg = (0, 0) 
sum_deg = 0 
degs = dict()
for k, v in adj_dict.items(): 
    deg = len(v)
    if deg < min_deg[0]: 
        min_deg = (deg, k)
    if deg > max_deg[0]: 
        max_deg = (deg, k)
    sum_deg += deg 

    if deg not in degs.keys(): 
        degs[deg] = 1
    else: 
        degs[deg] += 1

edges = sum_deg / 2
avg_deg = sum_deg / n_nodes

print(f"Average degree: {avg_deg}")
print(f"Min degree: {min_deg}")
print(f"Max degree: {max_deg}")
print(f"Number of edges: {edges}")
print(adj_dict["Q17053748"])

# import matplotlib.pyplot as plt

# plt.bar(list(degs.keys()), degs.values(), color='g')
# plt.show()
