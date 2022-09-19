# used to find the max number of objects in the ocid dataset.
# we use the number to set topk during evaluation.
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

from datasets import OCIDDataset
from tqdm import tqdm

ocid_dataset = OCIDDataset(image_set="test")
ocid_max_num_object = 0
#print(ocid_dataset.max_num_object)

for i in tqdm(ocid_dataset):
    if ocid_dataset.max_num_object > ocid_max_num_object:
        ocid_max_num_object = ocid_dataset.max_num_object
print("ocid max num object", ocid_max_num_object)

# The result is 20.