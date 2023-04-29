# subset the image vectors to only contain test/train data for the seen and unseen classes we are using for the project
import pickle

with open('data/image_vectors_emb.pkl', 'rb') as f:
    data = pickle.load(f)

subset_data = dict()

train_data = data['train_data']
test_data = data['test_data']

subset_data['train_data'] = []
subset_data['test_data'] = []

save_labels = ['northern_watersnake', 'boa_constrictor'] # , 'rough_greensnake'

for tensor, label in train_data:
    if label in save_labels:
        subset_data['train_data'].append((tensor, label))

for tensor, label in test_data:
    if label in save_labels:
        subset_data['test_data'].append((tensor, label))

with open("data/image_vectors_subset.pkl", "wb") as f:
    pickle.dump(subset_data, f)


# test the new dataset
# with open("image_vectors_subset.pkl", "rb") as f:
#     data = pickle.load(f)

# labels = []
#
# for item in data['train_data']:
#     if item[1] not in labels:
#         labels.append(item[1])
#
#
# for item in data['test_data']:
#     if item[1] not in labels:
#         labels.append(item[1])
#
# print(labels)