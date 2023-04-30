import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pickle

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-2-cd82c01330ab
'''

# node features (x)
with open("data/word_embedding_model/labels.pkl", "rb") as f:
    x = pickle.load(f)
    x = torch.from_numpy(x)
    x = x.float().to(device)
    # x = torch.t(x)

# edge index
with open("data/graph_struct/edge_index.pkl", "rb") as f:
    edge_index = pickle.load(f).to(device)


# load in the targets
with open("data/target.pkl", "rb") as f:
    targets = pickle.load(f).to(device)

data = Data(x = x, edge_index = edge_index, y = targets)


class GCN(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_input, 2048)
        self.conv2 = GCNConv(2048, 2048)
        self.conv3 = GCNConv(2048, 1024)
        self.conv4 = GCNConv(1024, 1024)
        self.conv5 = GCNConv(1024, 512)
        self.conv6 = GCNConv(512, n_output)

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        x = F.leaky_relu(self.conv1(x,edge_index),0.2)
        x = F.leaky_relu(self.conv2(x,edge_index),0.2)
        x = F.leaky_relu(self.conv3(x,edge_index),0.2)
        x = F.leaky_relu(self.conv4(x,edge_index),0.2)
        x = F.leaky_relu(self.conv5(x,edge_index),0.2)
        x = F.leaky_relu(self.conv6(x,edge_index),0.2)
        return F.log_softmax(x,dim=1) #edit this later

model = GCN(300, 2048).to(device)

lr = 1e-3
wd = 5e-4
optimizer_name = "Adam"
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 300

def train(model, data, optimizer, loss_fn):
    model.train()
    # Clear gradients
    optimizer.zero_grad()
    # Forward pass
    pred = model(data.x, data.edge_index)
    # Calculate loss function
    loss = loss_fn(pred, data.y)
    # Compute gradients
    loss.backward()
    # Tune parameters
    optimizer.step()

    return loss, pred

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []
accuracies = []
outputs = []

for epoch in range(1,epochs):
    loss, pred = train(model, data, optimizer, criterion)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f}% ')
