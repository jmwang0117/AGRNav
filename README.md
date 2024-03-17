<div align="center">   

# AGRNav: Efficient and Energy-Saving Autonomous Navigation for Air-Ground Robots in Occlusion-Prone Environments
</div>

## News

- [2024/01]: AGRNav is accepted to ICRA 2024.
- [2023/12]: We will release  [HE-Nav](https://github.com/jmwang0117/HE-Nav), a more efficient, energy-saving and ESDF-free navigation system.
- [2023/11]: The code for training [SCONet](https://github.com/jmwang0117/SCONet) is in another repository.
- [2023/10]: Our SCONet [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/ERcBqaBqJRtOm4biZ-nXRlUBAUq0AhdEwy4yagrD7ZCCow?e=vzmmEU) can be downloaded through OneDrive.
- [2023/09]: The [3D model ](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/ERX7ejbV3xdOkLQe5SMgGG0Bh6D1qGd-9vg5iMWpi8VQsw?e=H07haj) in the simulation environment can be downloaded in OneDrive.
- [2023/08]: 🔥 We released the code of AGRNav in the simulation environment.

</br>

If you find this work helpful, kindly show your support by giving us a free ⭐️. Your recognition is truly valued.

<p align = "center">
<img src="figs/sim1.gif" width = "700" height = "400" border="3" />
</p>

If you find this work useful in your research, please consider citing:
```
@inproceedings{jmwang,
  title={AGRNav: Efficient and Energy-Saving Autonomous Navigation for Air-Ground Robots in Occlusion-Prone Environments},
  author={Wang, Junming and and Sun, Zekai and Guan, Xiuxian and Shen, Tianxiang and Zhang, Zongyuan and Duan, Tianyang and Huang, Dong and Zhao, Shixiong and Cui, Heming},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={},
  year={2024},
  organization={IEEE}
}
```
## Installation
The code was tested with `python=3.6.9`, as well as `pytorch=1.10.0+cu111` and `torchvision=0.11.2+cu111`. 

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

1. Clone the repository locally:

```
 git clone https://github.com/jmwang0117/AGRNav.git
```
2. We recommend using **Docker** to run the project, which can reduce the burden of configuring the environment, you can find the Dockerfile in our project, and then execute the following command:
```
 docker build . -t skywalker_robot -f Dockerfile
```
3. After the compilation is complete, use our **one-click startup script** in the same directory:
```
 bash create_container.sh
```

 **Pay attention to switch docker image**

4. Next enter the container and use git clone our project
```
 docker exec -it robot bash
```
5. Then catkin_make compiles this project
```
 apt update && sudo apt-get install libarmadillo-dev ros-melodic-nlopt

```

6. Since need to temporarily save the point cloud, please modify the path in the following file:
```
/root/AGRNav/src/perception/launch/inference.launch

/root/AGRNav/src/perception/SCONet/network/data/SemanticKITTI.py

/root/AGRNav/src/perception/script/pointcloud_listener.py
```


## Run the following commands 
```
pip install pyyaml
pip install rospkg
pip install imageio
catkin_make
source devel/setup.bash
sh src/run.sh
```

You've begun this project successfully; **enjoy yourself!**


## Dataset

- [x] SemanticKITTI

## Acknowledgement

Many thanks to these excellent open source projects:
- [Prometheus](https://github.com/amov-lab/Prometheus)
- [LMSCNet](https://github.com/astra-vision/LMSCNet)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
- [Terrestrial-Aerial-Navigation](https://github.com/ZJU-FAST-Lab/Terrestrial-Aerial-Navigation)
- [Fast-Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner)

