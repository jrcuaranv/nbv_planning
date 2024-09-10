# nbv_planning
A Next-best-view (NBV) planning approach for target-aware active semantic mapping in horticultural environments.
Further description will be available soon.

<div align="center">
  <a href="https://youtu.be/l6IBl4n1GJ0">
    <img src="media/active_mapping.gif" height="250" alt="Active_mapping">
  </a>
</div>

## Dependencies
- ROS Noetic
- Nanoflann
- Opencv
- Sklearn
- scipy
- PCL (Point Cloud Library)
- OctoMap
- octomap_msgs
- octomap_rviz_plugins
## Bulding the project (Tested with ubuntu 20.04)
### Install dependencies
- Install ROS Noetic
- Opencv
```
sudo apt install libopencv-dev python3-opencv
```
- PCL
```
sudo apt-get install libpcl-dev
```
- Sklearn
```
pip3 install scikit-learn
```
- Scipy
```
pip3 install scipy==1.10.1

```
- Octomap
```
sudo apt install ros-noetic-octomap ros-noetic-octomap-server
ros-noetic-octomap-msgs
ros-noetic-octomap-rviz-plugins
```
- Nanoflann
1. Download [nanoflann](https://github.com/jlblancoc/nanoflann/releases/tag/v1.4.3)
2. Decompress the zip file
3. Install Nanoflann
```
cd nanoflann
cd ..
mkdir nanoflann-install
cd nanoflann
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=<path-to-nanoflann-install> ..
make install
```
### Clone and build nbv_planning and SSMI-agriculture projects
```
cd catkin_ws/src
git clone https://github.com/jrcuaranv/SSMI-agriculture.git
git clone https://github.com/jrcuaranv/nbv_planning.git
cd ..
catkin_make
```

### Run the mapping and planning pipeline
1. Go to nbv_planning/data/models and run script install_models.sh to copy gazebo plant models to your home directory.
```
cd catkin_ws/src/nbv_planning/data/models
./install_models.sh
```
2. Launch main files in independent terminals (easy for debugging and visualization)
```
cd catkin_ws
source devel/setup.bash
roslaunch nbv_planning gazebo_environment.launch
roslaunch semantic_octomap semantic_octomap_evaluation.launch
roslaunch nbv_planning nbv_planning.launch
```
3. If rviz does not start properly, open rviz and visualize main topics:
- Set world as fixed frame
- Add ColorOccupancyGrid and set /octomap_full as topic and cell_color as voxel coloring.
- Add /camera2/color/rgb and /camera2/color/semantics topics 
- Add Axes to visualize the /camera2_frame topic (the target goal)

