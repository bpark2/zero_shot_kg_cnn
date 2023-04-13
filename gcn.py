import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-2-cd82c01330ab
'''

data = ''#need to set data to the dataset

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

def train(model, data, optimizer, loss_fn):
    model.train()
    # Clear gradients# Clear gradients
    optimizer.zero_grad()
    # Forward pass
    label, emb = model(data.x, data.edge_index)
    # Calculate loss function
    loss = loss_fn(label, data.y)
    # Compute gradients
    loss.backward()
    # Tune parameters
    optimizer.step()

    pred = label.argmax(dim=1)
    acc = (pred == data.y).sum() / len(data.y)

    return loss, acc, pred

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []
accuracies = []
outputs = []

for epoch in range(1,epochs):
    loss, acc, pred = train(model, data, optimizer, criterion)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(pred)
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f}% | Acc: {acc*100:.2f}%')
