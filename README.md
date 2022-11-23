# Mean Shift Mask Transformer for Unseen Object Instance Segmentation

### Introduction
Segmenting unseen objects is a critical task in many different domains. For example, a robot may need to grasp an unseen object, which means it needs to visually separate this object from the background and/or other objects. Mean shift clustering is a common method in object segmentation tasks. However, the traditional mean shift clustering algorithm is not easily integrated into an end-to-end neural network training pipeline. In this work, we propose the Mean Shift Mask Transformer (MSMFormer), a new transformer architecture that simulates the von Mises-Fisher (vMF) mean shift clustering algorithm, allowing for the joint training and inference of both the feature extractor and the clustering. Its central component is a hypersphere attention mechanism, which updates object queries on a hypersphere. To illustrate the effectiveness of our method, we apply MSMFormer to Unseen Object Instance Segmentation, which yields a new state-of-the-art of 87.3 Boundary F-meansure on the real-world Object Clutter Indoor Dataset (OCID).

<p align="center"><img src="./data/pics/overview.png" width="850" height="524"/></p>

### Mean Shift Mask Transformer Architecture
<p align="center"><img src="./data/pics/model.png" width="850" height="463"/></p>

### Download

- Download our trained checkpoints from [here](https://drive.google.com/drive/folders/1lmmTLqlNlN4AjwzWT7lmPrMygQNS7FmR?usp=sharing).

### Citation

If you find Mean Shift Mask Transformer useful in your research, please consider citing:

```
@misc{https://doi.org/10.48550/arxiv.2211.11679,
  doi = {10.48550/ARXIV.2211.11679},
  
  url = {https://arxiv.org/abs/2211.11679},
  
  author = {Lu, Yangxiao and Chen, Yuqiao and Ruozzi, Nicholas and Xiang, Yu},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Mean Shift Mask Transformer for Unseen Object Instance Segmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

### Install



Some code is from [UCN](https://github.com/IRVLUTD/UnseenObjectClustering) and [Mask2Former](https://github.com/facebookresearch/Mask2Former).

### Training on the Tabletop Object Dataset (TOD)
1. Download the Tabletop Object Dataset (TOD) from [here](https://drive.google.com/uc?export=download&id=1Du309Ye8J7v2c4fFGuyPGjf-C3-623vw) (34G).

2. Create a symlink for the TOD dataset
    ```Shell
    cd $ROOT/data
    ln -s $TOD_DATA tabletop
    ```

3. Training and testing on the TOD dataset
    ```Shell
    cd $ROOT/MSMFormer

    # multi-gpu training, we used 4 GPUs
   python tabletop_train_net_pretrained.py --num-gpus 4

    ```


### Testing on the OCID dataset and the OSD dataset

1. Download the OCID dataset from [here](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/), and create a symbol link:
    ```Shell
    cd $ROOT/data
    ln -s $OCID_dataset OCID
    ```

2. Download the OSD dataset from [here](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/osd/), and create a symbol link:
    ```Shell
    cd $ROOT/data
    ln -s $OSD_dataset OSD
    ```
3. Test
    ```Shell
    cd $ROOT/lib/fcn
    python test_demo.py
    ```
   Or you can directly run $ROOT/lib/fcn/test_demo.py with IDE like PyCharm.

