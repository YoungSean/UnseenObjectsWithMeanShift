import numpy as np
from matplotlib import pyplot as plt
from tabletop_dataset import TableTopDataset
dataset = TableTopDataset()
# sample = dataset[3]
# img = sample["raw_image"]
# plt.imshow(img, interpolation='nearest')
# plt.show()
# depth = sample["raw_depth"]
# plt.imshow(depth, interpolation='nearest')
# plt.show()
#
# label = sample["labels"]
# plt.imshow(np.squeeze(label), interpolation='nearest')
# plt.show()
# print("done")
for i in range(3, 7):
    sample = dataset[i]
    img = sample["raw_image"]
    plt.imshow(img, interpolation='nearest')
    plt.show()
    depth = sample["raw_depth"]
    plt.imshow(depth[:,:,0], interpolation='nearest')
    plt.show()
    print("done")
#     print(sample.keys())