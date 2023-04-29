# preprocess the graph structure to be an edge index format for the GCN
import pickle
import json

# load in the adj dictionary
import torch

with open("data/graph_struct/graph_adj_dict.json", "r") as f:
    adj_dict = json.load(f)

with open("data/graph_struct/ids.json", "r") as f:
    ids = json.load(f)

# convert adj dict from ids to numeric
num_adj_dict = dict()
for k, v in adj_dict.items():
    num_k = ids.index(k)
    items = []
    for item in v:
        num_item = ids.index(item)
        items.append(num_item)
    num_adj_dict[num_k] = items

# create the e_to, e_from tensors
e_from = []
e_to = []
for k, v in num_adj_dict.items():
    for item in v:
        e_from.append(k)
        e_to.append(item)

# create the edge index tensor
edge_index = torch.tensor([e_from, e_to], dtype = torch.int)
print(edge_index)

# pickle that
with open("data/graph_struct/edge_index.pkl", "wb") as f:
    pickle.dump(edge_index, f)