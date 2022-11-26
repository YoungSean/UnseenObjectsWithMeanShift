# Mean Shift Mask Transformer for Unseen Object Instance Segmentation

### Introduction
Segmenting unseen objects is a critical task in many different domains. For example, a robot may need to grasp an unseen object, which means it needs to visually separate this object from the background and/or other objects. Mean shift clustering is a common method in object segmentation tasks. However, the traditional mean shift clustering algorithm is not easily integrated into an end-to-end neural network training pipeline. In this work, we propose the Mean Shift Mask Transformer (MSMFormer), a new transformer architecture that simulates the von Mises-Fisher (vMF) mean shift clustering algorithm, allowing for the joint training and inference of both the feature extractor and the clustering. Its central component is a hypersphere attention mechanism, which updates object queries on a hypersphere. To illustrate the effectiveness of our method, we apply MSMFormer to Unseen Object Instance Segmentation, which yields a new state-of-the-art of 87.3 Boundary F-meansure on the real-world Object Clutter Indoor Dataset (OCID).
[arXiv](https://arxiv.org/abs/2211.11679)
<p align="center"><img src="./data/pics/overview.png" width="850" height="524"/></p>

### Mean Shift Mask Transformer Architecture
<p align="center"><img src="./data/pics/model.png" width="850" height="463"/></p>


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
### Required Environment
- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above

### Install
The code is based on [Detetron2 framework](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html).
1. Install PyTorch
2. Install [Detetron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
3. Install other packages

For example, in an anaconda environment:
```Shell
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -r /path/to/requirement.txt
```
Some code is from [UCN](https://github.com/IRVLUTD/UnseenObjectClustering) and [Mask2Former](https://github.com/facebookresearch/Mask2Former). 
The main folder is **$ROOT/MSMFormer/meanshiftformer**. The Python classes begin with "pretrained" are used for Unseen Object Instance Segmentation.
The Hypersphere Attention is in [this file](https://github.com/YoungSean/UnseenObjectsWithMeanShift/blob/master/MSMFormer/meanshiftformer/modeling/transformer_decoder/attention_util.py).

### Download
- Download the pretrained backbone checkpoints from [UCN](https://github.com/IRVLUTD/UnseenObjectClustering). They are *seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth* and *seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth*. 
Then replace the file paths in this [script](https://github.com/YoungSean/UnseenObjectsWithMeanShift/blob/bbc0f52d8723caa2a168790dcc793dd2cc933f36/lib/fcn/get_network_crop.py).
- Download our trained checkpoints from [here](https://drive.google.com/drive/folders/1lmmTLqlNlN4AjwzWT7lmPrMygQNS7FmR?usp=sharing).

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
4. Demo images in $ROOT/data/demo
   For some real world images from the lab, an example python script is $ROOT/tools/test_image_with_ms_transformer.py.


