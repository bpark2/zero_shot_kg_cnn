import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
import numpy as np

# load in the data from file
dataset = pickle.load(open('data/image_vectors_subset.pkl', 'rb'))

# change the outputs from string labels to numeric

le = preprocessing.LabelEncoder()

def relabel_data(encoder, data):
    """

    Args:
        encoder: LabelEncoder for use translating string labels to numeric
        data: the data to convert the labels for

    Returns: a list of (tensor, int) tuples with the new numeric labels

    """
    tensors_labels = list(zip(*data))
    targets = le.fit_transform(tensors_labels[1])
    relabeled_data = [tensors_labels[0], targets]
    relabeled_data = list(zip(*relabeled_data))
    return relabeled_data

dataset['train_data'] = relabel_data(le, dataset['train_data'])
dataset['test_data'] = relabel_data(le, dataset['test_data'])

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
with open("data/target.pkl", "wb") as f:
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