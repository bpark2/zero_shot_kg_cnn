# get the test accuracy for the unseen class (rough greensnake: Q426062 - index: 644)
import pickle
import numpy as np
import torch

with open("data/image_vectors_unseen.pkl", "rb") as f:
    image_vectors = pickle.load(f)

with open("data/predicted_classifier_weights.pkl", "rb") as f:
    predicted_classifiers = pickle.load(f)

# extract the image vectors for the unseen class
unseen_test = []
for vect, _ in image_vectors['test_data']:
    unseen_test.append(vect)

# extract the predicted classifier
classifier_green = predicted_classifiers[644]

# classify images, if >= 0.5 classify as rough greensnake, if < 0.5 not rough greensnake
green_snakes = 0
for vect in unseen_test:
    prediction = torch.dot(vect, classifier_green)
    if prediction >= 0.5:
        green_snakes += 1

print(f"Accuracy: {green_snakes / len(unseen_test)}, number of correctly classified snakes: {green_snakes} / {len(unseen_test)}")

# on the actual training classes:
with open("data/image_vectors_seen.pkl", "rb") as f:
    image_vectors = pickle.load(f)

classifier_boa = predicted_classifiers[467]
classifier_water = predicted_classifiers[1465]

# extract the image vectors for the unseen class
test_boa = []
test_water = []
for vect, label in image_vectors['test_data']:
    if label == "boa_constrictor":
        test_boa.append(vect)
    if label == "northern_watersnake":
        test_water.append(vect)

def classify(vector):
    """Classify the image vector by returning the highest predicted class value.

    Args:
        vector: image vector to classify

    Returns:
        an integer representing the selected class:
         0 : boa constrictor
         1 : northern watersnake
         2 : rough greensnake
    """
    scores = []
    scores.append(torch.dot(vector, classifier_boa).detach().numpy())
    scores.append(torch.dot(vector, classifier_water).detach().numpy())
    scores.append(torch.dot(vector, classifier_green).detach().numpy())
    return np.argmax(scores)

def test_class(vectors, class_num):
    """Classify all images in a given class.

    Args:
        vectors: the list of image vectors to test
        class_num: the class number for the images

    Returns:
        the number of correct classifications
    """
    correct = 0
    for vect in vectors:
        pred_class = classify(vect)
        if pred_class == class_num:
            correct += 1
    return correct

boa_correct = test_class(test_boa, 0)
water_correct = test_class(test_water, 1)
green_correct = test_class(unseen_test, 2)

total_correct = boa_correct + water_correct + green_correct
total_shown = len(test_boa) + len(test_water) + len(unseen_test)
overall_acc = total_correct / total_shown

print(f"Boa performance: {boa_correct} / {len(test_boa)}")
print(f"Water performance: {water_correct} / {len(test_water)}")
print(f"Green performance: {green_correct} / {len(unseen_test)}")
print(f"Overall training acc: {overall_acc}")