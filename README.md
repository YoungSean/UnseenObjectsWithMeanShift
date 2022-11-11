# Mean Shift Mask Transformer for Unseen Object Instance Segmentation

### Introduction
Segmenting unseen objects is a critical task in many different domains. For example, a robot may need to grasp an unseen object, which means it needs to visually separate this object from the background and/or other objects. Mean shift clustering is a commonly-used method in object segmentation tasks. However, the traditional mean shift clustering algorithm is not easily integrated into an end- to-end neural network training pipeline. In this work, we propose the Mean Shift Mask Transformer (MSMFormer), a new transformer architecture that simulates the von Mises-Fisher (vMF) mean shift clustering algorithm, allowing for the joint inference of both the feature extractor and the trained clustering. Its central component is a hypersphere attention mechanism, which updates object queries on a hypersphere. To illustrate the effectiveness of our method, we apply MSMFormer to Unseen Object Instance Segmentation, which yields a new state-of-the-art of 87.3 Boundary F-meansure on the real-world Object Clutter Indoor Dataset (OCID).


### Download

- Download our trained checkpoints from [here](https://drive.google.com/drive/folders/1lmmTLqlNlN4AjwzWT7lmPrMygQNS7FmR?usp=sharing).



