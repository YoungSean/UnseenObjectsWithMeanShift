# Mean Shift Mask Transformer for Unseen Object Instance Segmentation

### Introduction
Segmenting unseen objects is a critical task in many different domains. For example, a robot may need to grasp an unseen object, which means it needs to visually separate this object from the background and/or other objects. Mean shift clustering is a common method in object segmentation tasks. However, the traditional mean shift clustering algorithm is not easily integrated into an end-to-end neural network training pipeline. In this work, we propose the Mean Shift Mask Transformer (MSMFormer), a new transformer architecture that simulates the von Mises-Fisher (vMF) mean shift clustering algorithm, allowing for the joint training and inference of both the feature extractor and the clustering. Its central component is a hypersphere attention mechanism, which updates object queries on a hypersphere. To illustrate the effectiveness of our method, we apply MSMFormer to Unseen Object Instance Segmentation, which yields a new state-of-the-art of 87.3 Boundary F-measure on the real-world Object Clutter Indoor Dataset (OCID).
[arXiv](https://arxiv.org/abs/2211.11679)
<p align="center"><img src="./data/pics/overview.png" width="797" height="523"/></p>

### Mean Shift Mask Transformer Architecture
<p align="center"><img src="./data/pics/model.png" width="624" height="340"/></p>


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
- Create a folder $ROOT/data/checkpoints
- Download the pretrained backbone checkpoints from [UCN](https://github.com/IRVLUTD/UnseenObjectClustering). They are *seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth* and *seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth*. 
Then move the checkpoint files into $ROOT/data/checkpoints. 
- Download our trained checkpoints from [here](https://drive.google.com/drive/folders/1lmmTLqlNlN4AjwzWT7lmPrMygQNS7FmR?usp=sharing). Then move the checkpoint files into $ROOT/data/checkpoints. (* ["RGB_*.pth"](https://drive.google.com/drive/folders/12HWlEXng-lmd9q-LGxJQp0SMaOu1FVcc?usp=sharing) files are checkpoints for RGB images. They can be ignored.)

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

### Demo
1. For Demo images in $ROOT/data/demo, you can run $ROOT/experiments/scripts/demo_msmformer_rgbd.sh to see the visual results. (* demo_msmformer_rgb.sh is only using RGB information.)
<p align="center"><img src="./data/pics/Figure_2.png" width="640" height="480"/> <img src="./data/pics/Figure_7.png" width="640" height="480"/></p>


2. An example python script is $ROOT/tools/test_image_with_ms_transformer.py.

   In terminal, run the following command:
   ```shell
   ./tools/test_image_with_ms_transformer.py  \
   --imgdir data/demo   \
   --color *-color.png   \
   --depth *-depth.png \
   --pretrained path/to/first/stage/network/checkpoint \
   --pretrained_crop path/to/second/stage/network/checkpoint \
   --network_cfg path/to/first/stage/network/config/file \
   --network_crop_cfg path/to/second/stage/network/config/file
   ```
   
   An example is shown as follows:
   ```shell
   ./tools/test_image_with_ms_transformer.py  \
   --imgdir data/demo   \
   --color *-color.png   \
   --depth *-depth.png \
   --pretrained data/checkpoints/norm_model_0069999.pth \
   --pretrained_crop data/checkpoints/crop_dec9_model_final.pth \
   --network_cfg MSMFormer/configs/tabletop_pretrained.yaml \
   --network_crop_cfg MSMFormer/configs/crop_tabletop_pretrained.yaml
   ```

### Running with ROS on a real camera for real-world unseen object instance segmentation

- Make sure our pretrained checkpoints are downloaded.

    ```Shell
    # start realsense if you use a realsense camera
    roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera

    # start rviz
    rosrun rviz rviz -d ./ros/segmentation.rviz

    # run segmentation on a realsense camera, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_seg_transformer_test_segmentation_realsense.sh $GPU_ID
    
    # run segmentation on a Fetch camera, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh $GPU_ID
    
    ```
