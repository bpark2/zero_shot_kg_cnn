# import statements
import numpy as np
import pickle
from PIL import Image
import os
from tqdm.auto import tqdm
import json

# data loading functions from source 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_data(input_file):
    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.dstack((x[:, :4096], x[:, 4096:8192], x[:, 8192:]))
    x = x.reshape((x.shape[0], 64, 64, 3))

    return x, y

# make a mapping dictionary 
num_cls = dict() # number (y-label) to WordNet id
cls_num = dict() # WordNet id to number (y-label)
num_wd = dict() # number (y-label) to Wikidata ID
labels = [] # list of all Wikidata IDs

with open("map_clsloc.txt", "r") as map: 
  for line in map: 
    ids = line.split(" ")
    wnid = ids[0]
    num_id = ids[1]
    num_cls[num_id] = wnid
    cls_num[wnid] = num_id

with open("mapping.json", "r") as map2: 
  map_dict = json.load(map2)
  for k,v in map_dict.items(): 
    wdid = v.split("/")[-1]
    num = cls_num[k]
    num_wd[num] = wdid
    labels.append(wdid)

# loop to create directories by label
import os 
os.mkdir("train/")
for id in labels: 
  os.mkdir(f"train/{id}/")

for batch in range(1,11): 
  x, y = load_data(f"train_data_batch_{batch}")
  print(f"Reconstructing images from batch {batch} ... ")
  for i in tqdm(range(len(x))): 
    lbl = num_wd[str(y[i])]
    img = Image.fromarray(x[i])
    img.save(f"train/{lbl}/img_{(batch * len(x)) + i}.jpg", "JPEG")
  print("Reconstruction complete.")