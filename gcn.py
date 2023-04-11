import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-2-cd82c01330ab
'''

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 2048)
        self.conv2 = GCNConv(2048, 2048)
        self.conv3 = GCNConv(2048, 1024)
        self.conv4 = GCNConv(1024, 1024)
        self.conv5 = GCNConv(1024, 512)
        self.conv6 = GCNConv(512, int(data.num_classes))

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.leaky_relu(self.conv1(x,edge_index),0.2)
        x = F.leaky_relu(self.conv2(x,edge_index),0.2)
        x = F.leaky_relu(self.conv3(x,edge_index),0.2)
        x = F.leaky_relu(self.conv4(x,edge_index),0.2)
        x = F.leaky_relu(self.conv5(x,edge_index),0.2)
        x = F.leaky_relu(self.conv6(x,edge_index),0.2)
        return F.log_softmax(x,dim=1) #edit this later

model = GCN().to(device)

lr = 1e-3
wd = 5e-4
optimizer_name = "Adam"
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 300

def train():
    model.train()
    optimizer.zero_grad()
    m = 2
    loss = 0
    for i in range(m):
        loss += F.mse_loss(data.train_mask, data.y[data.train_mask]).backward()
    loss /= m
    optimizer.step()

for epoch in range(1,epochs):
    print("AHHHH")
# @torch.no_grad()
# def test():
#     model.eval()
#     logits = model()
