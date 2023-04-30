import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pickle
from pathlib import Path
import sys

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-2-cd82c01330ab
'''

def load_targets(target_dirname):
    train_indices = []
    target_dir = Path(target_dirname)
    files = target_dir.glob("*.pkl")
    first = True
    for file in files:
        target_id = int(file.name.split("_")[0])
        with file.open("rb") as f:
            classifier = pickle.load(f)
        if first:
            targets = classifier
            first = False
        else:
            targets = torch.cat((targets, classifier), axis = 0)
        train_indices.append(target_id)
    return targets, train_indices

# node features (x)
with open("data/word_embedding_model/labels.pkl", "rb") as f:
    x = pickle.load(f)
    x = torch.from_numpy(x)
    x = x.float().to(device)

# edge index
with open("data/graph_struct/edge_index.pkl", "rb") as f:
    edge_index = pickle.load(f).to(device)

# load in the targets
targets, train_indices = load_targets("data/targets")

data = Data(x = x, edge_index = edge_index, y = targets)

class ModifiedMSELoss(torch.nn.Module):
    """Implements a custom MSE-based loss function for the zero-shot visual classifier learning.
    """
    def __init__(self):
        super().__init__()
        self.__name__ = "Modified MSE Loss"

    def forward(self, inputs, targets, train_indices):
        """
        Args:
            inputs: the tensor of predicted classifiers
            targets: the tensor of target classifiers
            train_indices: A list of the indices for which we have targets, serves as a mapping from the
                targets tensor to the inputs tensor indices

        Returns:
            the custom MSE loss calculated based only on the classifiers for which we have training data
        """
        # for each of the targets we have, get the mse loss and add to total loss
        mse = 0
        num_train = len(train_indices)
        for i in range(num_train):
            idx = train_indices[i]
            predicted_classifier = inputs[idx]
            target_classifier = targets[i]
            mse += F.mse_loss(predicted_classifier, target_classifier)

        # divide by the number of training targets
        mse /= num_train

        # return the value
        return mse

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
        x = F.log_softmax(x, dim = 1)
        return F.normalize(x, p = 2)

model = GCN(300, 2048).to(device)

lr = 1e-3
wd = 5e-4
optimizer_name = "Adam"
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 5

def train(model, data, optimizer, loss_fn):
    model.train()
    # Clear gradients
    optimizer.zero_grad()
    # Forward pass
    pred = model(data.x, data.edge_index)
    # Calculate loss function
    loss = loss_fn(pred, data.y, train_indices)
    # Compute gradients
    loss.backward()
    # Tune parameters
    optimizer.step()

    return loss, pred

criterion = ModifiedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []
accuracies = []
outputs = []

for epoch in range(1,epochs):
    loss, pred = train(model, data, optimizer, criterion)
    losses.append(loss)
    print(f'Epoch {epoch:>3} | Loss: {loss} ')

# get the final predicted model weights
pred = model(data.x, data.edge_index)

# save the predicted classifier weights
with open("data/predicted_classifier_weights.pkl", "wb") as f:
    pickle.dump(pred, f)