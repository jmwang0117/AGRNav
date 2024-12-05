<div align="center">   

# 🤖 AGRNav: Efficient and Energy-Saving Autonomous Navigation for Air-Ground Robots in Occlusion-Prone Environments

</div>

## 🎉 Chinese Media Reports
* [AMOV Lab Research Scholarship](https://mp.weixin.qq.com/s/AXbW3LDgsl9knQBIMwIpvA) -- 2024.11: 5000 RMB
* [AMOV Lab Research Scholarship](https://mp.weixin.qq.com/s/PUwY04sMpVmz30kSn6XdzQ) -- 2024.10: 5000 RMB
  
## 🤗 AGR-Family Works

* [OMEGA](https://jmwang0117.github.io/OMEGA/) (RA-L 2024.12): The First AGR-Tailored Dynamic Navigation System.
* [HE-Nav](https://jmwang0117.github.io/HE-Nav/) (RA-L 2024.09): The First AGR-Tailored ESDF-Free Navigation System.
* [AGRNav](https://github.com/jmwang0117/AGRNav) (ICRA 2024.01): The First AGR-Tailored Occlusion-Aware Navigation System.


## 📢 News
- [2024/01]: AGRNav is accepted to ICRA 2024.
- [2023/11]: The code for training [SCONet](https://github.com/jmwang0117/SCONet) is in another repository.
- [2023/09]: The [3D model ](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/ERX7ejbV3xdOkLQe5SMgGG0Bh6D1qGd-9vg5iMWpi8VQsw?e=H07haj) in the simulation environment can be downloaded in OneDrive.
- [2023/08]: 🔥 We released the code of AGRNav in the simulation environment.

</br>

If you find this work helpful, kindly show your support by giving us a free ⭐️. Your recognition is truly valued.

<p align = "center">
  <img src="figs/sim1.gif" width = "400" height = "260" border="1" style="display:inline;"/>
  
</p>

If you find this work useful in your research, please consider citing:
```
@INPROCEEDINGS{wang2024agrnav,
  author={Wang, Junming and Sun, Zekai and Guan, Xiuxian and Shen, Tianxiang and Zhang, Zongyuan and Duan, Tianyang and Huang, Dong and Zhao, Shixiong and Cui, Heming},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={AGRNav: Efficient and Energy-Saving Autonomous Navigation for Air-Ground Robots in Occlusion-Prone Environments}, 
  year={2024},
  pages={11133-11139}
}
```
## 🛠️ Installation
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


4. Next enter the container and use git clone our project
```
 docker exec -it robot bash
```

5. Re-clone the repository locally

```
 git clone https://github.com/jmwang0117/AGRNav.git
```

6. Since need to temporarily save the point cloud, please check the path in the following file:
```
/root/AGRNav/src/perception/launch/inference.launch

/root/AGRNav/src/perception/SCONet/network/data/SemanticKITTI.py

/root/AGRNav/src/perception/script/pointcloud_listener.py
```

7. SCONet pre-trained model is in the folder below:
```
/root/AGRNav/src/perception/SCONet/network/weights
```
8. If you want to use our 3D AGR model, please download the AGR model to the folder below:
```
/root/AGRNav/src/uav_simulator/Utils/odom_visualization/meshes
```

And modify the code on line 503 in the following file to AGR.dae
```
/root/AGRNav/src/uav_simulator/Utils/odom_visualization/src/odom_visualization.cpp
```

9. Run the following commands 
```
catkin_make
source devel/setup.bash
sh src/run.sh
```

You've begun this project successfully; **enjoy yourself!**


## 💽 Dataset

- [x] SemanticKITTI



## 🏆 Acknowledgement

Many thanks to these excellent open source projects:
- [Prometheus](https://github.com/amov-lab/Prometheus)
- [LMSCNet](https://github.com/astra-vision/LMSCNet)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
- [Terrestrial-Aerial-Navigation](https://github.com/ZJU-FAST-Lab/Terrestrial-Aerial-Navigation)
- [Fast-Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner)

