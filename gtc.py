import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np

torch.manual_seed(42)

TARGET = "boa_constrictor" # alt: boa_constrictor
TARGET_IDX = 467 # 467 for boa

# load in the data from file
dataset = pickle.load(open('data/image_vectors_subset.pkl', 'rb'))

# change the outputs from string labels to numeric
def relabel_data(data, target):
    """
    Relabel the data so that only data with the specified target label is 1, else 0.
    Args:
        data: data to relabel
        target: class to label as 1

    Returns:
        relabelled data, in [(vector, label), .., (vector, label)] format
    """
    tensors_labels = list(zip(*data))
    new_labels = []
    for label in tensors_labels[1]:
        if label == target:
            new_labels.append(1)
        else:
            new_labels.append(0)
    relabelled_data = [tensors_labels[0], new_labels]
    relabelled_data = list(zip(*relabelled_data))
    return relabelled_data


dataset['train_data'] = relabel_data(dataset['train_data'], TARGET)
dataset['test_data'] = relabel_data(dataset['test_data'], TARGET)

# load train and test data samples into dataloader
batch_size = 32
train_loader = DataLoader(dataset=dataset["train_data"], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset["test_data"], batch_size=batch_size, shuffle=False)

# build custom module for logistic regression
class LogisticRegression(torch.nn.Module):
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# create a model per class

# instantiate the model
n_inputs = 2048 # size of image feature vector
n_outputs = 1 # number of possible output classes
log_regr = LogisticRegression(n_inputs, n_outputs)

# defining the optimizer
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.001)
# defining Binary Cross-Entropy loss
criterion = torch.nn.BCELoss()

epochs = 50
Loss = []
acc = []
for epoch in range(epochs):
    for i, (vects, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = log_regr(vects)
        loss = criterion(outputs, labels.unsqueeze(-1).float())
        # Loss.append(loss.item())
        loss.backward()
        optimizer.step()
    Loss.append(loss.item())
    correct = 0
    for vects, labels in test_loader:
        outputs = log_regr(vects)
        predicted = torch.squeeze(outputs).round().detach().numpy()
        correct += np.sum((predicted == labels.detach().numpy()))
    accuracy = 100 * (correct.item()) / len(dataset['test_data'])
    acc.append(accuracy)
    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))

model_weights = log_regr.parameters()
classifier = list(model_weights)[0].data

# save the classifier
with open(f"data/targets/{TARGET_IDX}_target_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

plt.plot(Loss)
plt.xlabel("no. of epochs")
plt.ylabel("total loss")
plt.title("Loss")
plt.show()

plt.plot(acc)
plt.xlabel("no. of epochs")
plt.ylabel("total accuracy")
plt.title("Accuracy")
plt.show()