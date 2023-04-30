# subset the image vectors to only contain test/train data for the seen and unseen classes we are using for the project
import pickle

with open('data/image_vectors_emb.pkl', 'rb') as f:
    data = pickle.load(f)

subset_data = dict()

train_data = data['train_data']
test_data = data['test_data']

def subset_by_label(labels, fileout):
    subset_data['train_data'] = []
    subset_data['test_data'] = []

    for tensor, label in train_data:
        if label in labels:
            subset_data['train_data'].append((tensor, label))

    for tensor, label in test_data:
        if label in labels:
            subset_data['test_data'].append((tensor, label))

    with open(fileout, "wb") as f:
        pickle.dump(subset_data, f)


seen_labels = ['northern_watersnake', 'boa_constrictor']
unseen_labels = ['rough_greensnake']

subset_by_label(seen_labels, "data/image_vectors_seen.pkl")
subset_by_label(unseen_labels, "data/image_vectors_unseen.pkl")

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