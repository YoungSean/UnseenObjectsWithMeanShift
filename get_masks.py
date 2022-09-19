import torch
import numpy as np
#
#
#
# def combine_masks2(gt_masks):
#     """
#     Combine several bit masks [N, H, W] into a mask [H,W],
#     e.g. 8*480*640 tensor becomes a numpy array of 480*640.
#     [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
#     """
#     mask = gt_masks.to('cpu').numpy()
#     num, h, w = mask.shape
#     bin_mask = np.zeros((h, w))
#     num_instance = len(mask)
#     # if there is not any instance, just return a mask full of 0s.
#     if num_instance == 0:
#         return bin_mask
#
#     for m, object_label in zip(mask, range(1, 1+num_instance)):
#         label_pos = np.nonzero(m)
#         bin_mask[label_pos] = object_label
#     # filename = './bin_masks/001.png'
#     # cv2.imwrite(filename, bin_mask)
#     return bin_mask
#
# x = torch.tensor([[[1,0],
#                    [0,1]],
#                   [[0,1],[0,0]]])
# y = x.bool()
#
# print(combine_masks(y))

height = 2
width = 3
masks = [torch.tensor([[0,0,1], [0,0,1]]),
         torch.tensor([[1,0,1], [1,0,0]])]

raw_mask = torch.zeros((height, width), dtype=torch.uint8)
for i, m in enumerate(masks):
    raw_mask = torch.maximum(raw_mask, m * (i + 1))

print(raw_mask)