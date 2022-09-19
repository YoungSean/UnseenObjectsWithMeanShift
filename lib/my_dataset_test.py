
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.data import MetadataCatalog

#dataset = TableTopObject2(image_set='train')
#print(dataset[2])
from detectron2.data import DatasetCatalog


# def getTabletopDataset():
#     dataset = TableTopDataset(image_set='train')
#     print(len(dataset))
#     dataset_dicts = []
#     for i in range(len(dataset)):
#         dataset_dicts.append(dataset[i])
#
#     return dataset_dicts
print("before register")
DatasetCatalog.register("my_dataset", getTabletopDataset)
print("after register")
# later, to access the data:
#data = DatasetCatalog.get("my_dataset")
#print(len(data))


MetadataCatalog.get("my_dataset").thing_classes = ['__background__', 'object']

import random
import cv2
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
print("wait meta data")
metadata = MetadataCatalog.get("my_dataset")
print("meta done")
# for d in random.sample(data, 10):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)

#     #plt.imshow(out.get_image())
#     window_name = 'image'
#     cv2.imshow(window_name, out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
# #closing all open windows
# cv2.destroyAllWindows()
print("done!")