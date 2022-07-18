# Physically-based Editing of Indoor Scene Lighting from a Single Image

Zhengqin Li, Jia Shi, Sai Bi, Rui Zhu, Kalyan Sunkavalli, Miloš Hašan, Zexiang Xu, Ravi Ramamoorthi, Manmohan Chandraker

![](Images/teaser.png)

## Related links:
* [Trained models](https://drive.google.com/drive/folders/1jIaDIKKf3R_EpeMrxobA_HOMWO8D3W0C?usp=sharing)
* [Object insertion](https://github.com/lzqsd/VirtualObjectInsertion)
* [Dataset](https://ucsd-openrooms.github.io/)
* [Arxiv](https://arxiv.org/abs/2205.09343)

## Dependencies
We highly recommend using Anaconda to manage python packages. Required dependencies include:
* [pytorch](https://pytorch.org/)
* [pytorch3D](https://pytorch3d.org/)
* [OpenCV](https://opencv.org/)
* [Open3D](http://www.open3d.org/)
* [Nvidia Optix 5.11 - 6.0](https://developer.nvidia.com/designworks/optix/downloads/legacy)

## Train and test models on the OpenRooms dataset

![](Images/pipeline.png)

1. Download the [OpenRooms dataset](https://ucsd-openrooms.github.io/). 
2. Compile Optix-based shadow renderer with python binding. 
      * Go to [OptixRendererShadow](OptixRendererShadow) directory. Compile the code following this [link](https://github.com/lzqsd/OptixRenderer.git). 
4. Modify the pytorch3D code to support RMSE chamfer distance loss.
      * Go to [chamfer.py](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py). 
      * Add flag `isRMSE = False` to function (chamfer_distance)[https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py#:~:text=def-,chamfer_distance,-(]
