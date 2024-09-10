#!/usr/bin/env python
# It works with gazebo simulator, kinect free-moving camera
# v.2.2
from re import S
import rospy
from gazebo_msgs.msg import LinkState, LinkStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32, String
import numpy as np
import math
from scipy.spatial.transform import Rotation
import time
import os 
import cv2
from cv_bridge import CvBridge, CvBridgeError
import random
import copy
import tf2_ros
import tf
import geometry_msgs.msg
from semantic_octomap.srv import *


from utils_nbv_planning.TravellingSalesman import TravellingSalesman 
from utils_nbv_planning.utils_evaluation import compute_surface_coverage
from utils_nbv_planning.utils_active_mapping import ViewpointEvaluation
from utils_nbv_planning.utils import compute_xyz_vector, ros_pose_to_SE3, SE3_to_ros_pose
from utils_nbv_planning.utils_data import load_gt_data
from utils_nbv_planning.hsi_color_segmentation import get_semantic_image
import yaml

class NBVPlanner:
    def __init__(self):
        
        rospy.init_node('set_link_state', anonymous=True)

        try:
            os.remove("/home/jose/semantic_points.txt") # file written by semantic octomap
        except:
            print("semantic_points.txt does not exist")

        self.crop_size = rospy.get_param('/cam/crop_size') # size of images sent to octomap
        self.sampling_r = rospy.get_param('/sampling_radius') # distance from targets to camera
        self.cluster_size = rospy.get_param('/cluster_size') #expected fruit cluster size
        self.plant_height = rospy.get_param('/plant_height') # average plant height
        self.add_seg_noise = rospy.get_param('/add_seg_noise') # add segmentation noise
        self.output_dir = rospy.get_param('/octomap/save_path')
        self.data_dir = rospy.get_param('/data_dir')
        self.pointcloud_path = rospy.get_param('/octomap/save_path') + 'semantic_points.txt' #semantic nodes saved by sem. octomap
        self.fx = rospy.get_param('/cam/intrinsics/fx')
        self.fy = rospy.get_param('/cam/intrinsics/fy')
        # self.cx = rospy.get_param('/cam/intrinsics/cx')
        # self.cy = rospy.get_param('/cam/intrinsics/cy')
        self.cx, self.cy = self.crop_size/2 , self.crop_size/2
        
        self.K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1.0]])
        
        # transform from cam optical frame to camlink
        self.T_camlink_camframe = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])
        self.gt_pose_w_camlink  = None #ground truth pose of the camera link
        self.bridge = CvBridge()
        # Publisher
        self.pub_link_state = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=10)
        self.pub_semantic_image = rospy.Publisher('/camera2/color/semantics', Image, queue_size = 10)
        self.pub_rgb_image = rospy.Publisher('/camera2/color/rgb', Image, queue_size = 10)
        self.pub_depth_image = rospy.Publisher('/camera2/color/depth', Image, queue_size = 10)
        self.pub_reset_octomap = rospy.Publisher('/reset_octomap', Float32, queue_size = 10)

        # Subscriber
        rgb_topic = rospy.get_param("/cam/rgb_topic")
        depth_topic = rospy.get_param("/cam/depth_topic")
        rospy.Subscriber(rgb_topic, CompressedImage, self.callback_image_raw)
        rospy.Subscriber(depth_topic, Image, self.callback_depth_topic)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.callback_link_states)
        rospy.Subscriber("/semantic_centroids", PointCloud2, self.callback_semantic_centroids)
        rospy.Subscriber("/octomap_status", Float32, self.callback_octomap_status)

        self.cv_image = None
        self.cv_depth_image = None
        self.save_image_flag = False
        self.octomap_status = None
        
        self.map_res = rospy.get_param('/octomap/resolution') # has to be consistent with the octomap parameters
        phi = rospy.get_param('/octomap/phi')
        psi = rospy.get_param('/octomap/psi')
       
        
        self.tf_listener = tf.TransformListener()
        
        self.world_frame_id = 'world'
        
        self.gt_pose_w_camlink  = None #ground truth pose of the camera link

        self.T_camlink_camframe = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

        time.sleep(1)
        
        self.plant_ids = [3,4,5,6,7,8]
        self.gt_centroids, self.xyz_plant_origin, self.gt_pointcloud_path = None, None, None
        self.sem_centroids = []
        self.entropy = [] # just for evaluation 
        self.surface_coverage = []# just for evaluation 
        self.previous_surface_coverage = 0
        # self.score_methods = ['MI','UVC', 'UVPC', 'OAE', 'AE', 'OSAMCEP', 'RS']
        self.score_methods = ['OSAMCEP']
        self.score_method = None
        self.viewpoint_count  = 0
        self.initial_pose_id = 0
        self.max_viewpoints = None
        ###############################

        rospy.wait_for_service('querry_RLE')
        try:
            self.RLE_query = rospy.ServiceProxy('querry_RLE', GetRLE, persistent=True)
        except rospy.ServiceException as e:
            print("RLE Service initialization failed: %s"%e)
        
        self.vp_evaluation = ViewpointEvaluation(self.RLE_query, phi, psi)
        rate = rospy.Rate(0.5)
        print("Testing NBV planner")
        
                
        self.run_mapping_multiple_plants()
        print("done")
        while not rospy.is_shutdown():
            rate.sleep()
    
    def run_mapping_multiple_plants(self):
        
        self.max_viewpoints = 30

        #initial "warmup" for octomap
        self.octomap_warmap()
        self.pub_reset_octomap.publish(Float32(1))
        time.sleep(5)
        # end warmup

        
        for plant_id in self.plant_ids:
            self.plant_id = plant_id
            self.gt_centroids, self.xyz_plant_origin, self.gt_pointcloud_path = load_gt_data(plant_id, self.data_dir)
            self.pointcloud_gt = np.loadtxt(self.gt_pointcloud_path)
            plant_centroid = copy.deepcopy(self.xyz_plant_origin)
            plant_centroid[2] = self.plant_height/2
            # camlink poses for different initializations
            init_camlink_poses, _ = self.gen_cam_poses(0, 90, 30, 120, centroid=plant_centroid, theta_n_grid=5, phi_n_grid = 2, r=self.sampling_r)    


            for method in self.score_methods:
                np.random.seed(0)
                for i, pose in enumerate(init_camlink_poses):
                    print("Current Plant:", plant_id, " pose:", i, " Method:", method)
                    self.score_method = method
                    self.previous_surface_coverage = 0
                    self.initial_pose_id = i
                    self.viewpoint_count = 0
                    self.entropy = []
                    self.surface_coverage = []
                    self.execute_single_camlink_goal(pose) # initial scanning (single view)
                    self.evaluation()
                    self.nbv_planner()
                    self.pub_reset_octomap.publish(Float32(1))
                    time.sleep(10)

    def callback_octomap_status(self, msg):
        self.octomap_status = msg.data

    def callback_semantic_centroids(self,msg):
        latest_centroids = []
        for p in pc2.read_points(msg, skip_nans=True):
            latest_centroids.append([p[0], p[1], p[2]])  # Extract x, y, z coordinates
        
        # updating self.sem_centroids when new centroids are computed
        if len(latest_centroids)>0:
            self.sem_centroids = copy.deepcopy(latest_centroids)

    def callback_link_states(self, data):
        self.gt_pose_w_camlink = data.pose[0] # 0 corresponds to link_kinect
        

    def callback_depth_topic(self, data):
        try:
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(data)
            self.cv_depth_image = self.crop_center_square(self.cv_depth_image, self.crop_size)
            
        except Exception as e:
            rospy.logerr("Error converting depth Image to cv2: %s", e)
            return
    
    def callback_image_raw(self, data):
        try:
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
            self.cv_image = self.crop_center_square(self.cv_image, self.crop_size)
        except Exception as e:
            rospy.logerr("Error converting compressed image to cv2: %s", e)
            return
    
    def crop_center_square(self, image, size):
        h, w = image.shape[:2]
        top = (h - size)//2
        bottom = top + size
        left = (w - size) // 2
        right = left + size
        cropped_img = image[top:bottom, left:right]
        return cropped_img
    
    def nbv_planner(self):
        n_best_vp = 1 # number of best viewpoints to execute per centroid
        min_score = 0 # minimum score to achieve for each centroid
        complete_centroids = []
        
        while (self.viewpoint_count < self.max_viewpoints):
            print("Running NBV planner")
            if len(self.sem_centroids)>0:
                best_viewpoints = []
                for centroid in self.sem_centroids:
                    if centroid not in complete_centroids:
                        sorted_viewpoints, sorted_scores = self.best_viewpoint_for_ROI(centroid)
                        print("Best scores:", sorted_scores[0:3])
                        if sorted_scores[0] <= min_score:
                            complete_centroids.append(centroid)
                            print("New centroid completed:", centroid)
                        else:
                            for i in range (n_best_vp):
                                best_viewpoints.append(sorted_viewpoints[i])
        
                # Execute the best viewpoints following salesperson strategy
                if len(best_viewpoints) == 0:
                    print("No best viewpoints were found")
                    self.viewpoint_count += 1
                    self.evaluation()
                    continue

                vp_indices = self.run_salesman_algorithm(best_viewpoints)
                print("Executing ", len(vp_indices), " viewpoints")
                count_vp = 0
                for ind in vp_indices:
                    count_vp += 1
                    print("vp: ", count_vp, "/", len(vp_indices))
                    viewpoint_camlink = best_viewpoints[ind]
                    self.execute_single_camlink_goal(viewpoint_camlink)
                    self.evaluation()
                    

    def octomap_warmap(self):
        print("Octomap_warmap...")
        plant_centroid = [0,0,0.5]
        camlink_poses, camframe_poses = self.gen_cam_poses(0, 90, 30, 120, centroid=plant_centroid, theta_n_grid=2, phi_n_grid = 1, r=self.sampling_r)
        for pose in camlink_poses:
            print("Octomap warmap...")
            self.execute_single_camlink_goal(pose)

        
    def execute_single_camlink_goal(self, pose):
        max_time_octomap = 4
        self.publish_link_pose(pose,'link_kinect')
        time.sleep(1)
        image2 = copy.deepcopy(self.cv_image)
        depth_image2 = copy.deepcopy(self.cv_depth_image)
        depth_image2 = np.where(np.isnan(depth_image2), 3.0, depth_image2)
        current_pose = copy.deepcopy(self.gt_pose_w_camlink)
        self.octomap_status = None
        self.publish_main_topics(image2, depth_image2, current_pose)
        wait_time = 0
        while (self.octomap_status==None and wait_time < max_time_octomap):
            wait_time += 0.3
            time.sleep(0.3) # to update octomap
        print("viewpoint count:", self.viewpoint_count)
        self.viewpoint_count += 1
        
    def run_salesman_algorithm(self, viewpoints):
        points = []
        for vp in viewpoints:
            point = [vp.position.x, vp.position.y, vp.position.z]
            points.append(point)
        points = np.array(points)

        n_points = points.shape[0]
        D_ij = np.zeros((n_points, n_points))
        for i in range (points.shape[0]):
            for j in range (points.shape[0]):
                D_ij[i,j] = np.linalg.norm(points[i]-points[j])

        T = np.arange(n_points)
        TS = TravellingSalesman(T,D_ij,False)
        TS.runTS()
        path_indices = TS.T.tolist()
        return path_indices

    def evaluation(self):
        H = self.vp_evaluation.compute_entropy(self.gt_centroids, map_res = self.map_res, bbox_size=self.cluster_size)
        
        self.entropy.append(H)
        file_suffix = '_plant' + str(self.plant_id) + '_pose' + str(self.initial_pose_id) + '_' + self.score_method + '.txt'
        np.savetxt(self.output_dir + 'entropy' + file_suffix, np.array(self.entropy), fmt='%.8f')
        try:
            data_map = np.loadtxt(self.pointcloud_path).reshape(-1,3) #output of the mapping system
            surface_coverage = compute_surface_coverage(self.pointcloud_gt, data_map, offset = self.xyz_plant_origin, dist_threshold=self.map_res)
            np.savetxt(self.output_dir + 'pointcloud' + file_suffix, data_map, fmt='%.8f')
        except:
            surface_coverage = self.previous_surface_coverage
        
        self.surface_coverage.append(surface_coverage)
        self.previous_surface_coverage = surface_coverage
        try:
            os.remove(self.pointcloud_path) # file written by semantic octomap after each update
        except:
            print("semantic_points.txt does not exist")
        
        np.savetxt(self.output_dir + 'surface_coverage' + file_suffix, np.array(self.surface_coverage), fmt='%.8f')
        
        print("current plant model:", file_suffix)
        print("Current entropy:", H)
        print("Current surf. coverage:", surface_coverage)

    def best_viewpoint_for_ROI(self, centroid):
        r = self.sampling_r
        box_size = self.cluster_size # 3D bounding box enclosing the centroid
        sample_width = self.map_res # [m] how fine we want the sampling of rays
        camlink_poses, camframe_poses = self.gen_cam_poses(0, 360, 30, 150, centroid, theta_n_grid=10, phi_n_grid=5, r=r)
        scores = []
        box_size_on_frame = self.fx*box_size/r
        max_range = r + box_size/2
        
        skip_pixel = int(sample_width * self.fx/r)
        xyz_vector = compute_xyz_vector(box_size_on_frame, box_size_on_frame, self.fx, self.fy, skip_pixel=skip_pixel, max_range=max_range)
        for pose in camframe_poses:
            T = ros_pose_to_SE3(pose)
            score = self.vp_evaluation.compute_viewpoint_score(T, xyz_vector, centroid, self.score_method)
            scores.append(score)
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)[::-1].tolist()
        sorted_scores = scores[sorted_indices]
        sorted_camlink_poses = [camlink_poses[i] for i in sorted_indices]

        return sorted_camlink_poses, sorted_scores

    def publish_link_pose(self, pose, link_name):
        link_state_msg = LinkState()
        link_state_msg.link_name = link_name  # Replace with the name of your link
        link_state_msg.pose = pose  # Set the desired pose
        # link_state_msg.twist = Twist()  # Set the desired twist
        self.pub_link_state.publish(link_state_msg)

    def gen_cam_poses(self, theta_start, theta_end, phi_start, phi_end, centroid, theta_n_grid=5, phi_n_grid=5, r = 1): # values in degrees
        '''
        It generates viewpoints around a centroid, distributed on a sphere with radius r
        Theta and phi in degrees. n_greed determines the number of samples for Theta and phi to create a grid
        '''
        theta_list = ((np.pi/180)*np.linspace(theta_start, theta_end, theta_n_grid)).tolist()
        phi_list = ((np.pi/180)*np.linspace(phi_start, phi_end, phi_n_grid)).tolist()
        #https://mathworld.wolfram.com/SphericalCoordinates.html
        
        poses_cam_link = [] # poses of camera2_link (that we can control)
        poses_camframe = [] # poses of camera frame wrt world frame
        for phi in phi_list:
            for theta in theta_list:
                x = r*math.cos(theta)*math.sin(phi)
                y = r*math.sin(theta)*math.sin(phi)
                z = r*math.cos(phi)
                t_pg = np.array([[x],[y],[z]])
                R_pg = Rotation.from_euler('xyz',[np.pi,phi+np.pi/2,theta]).as_matrix()
                T_pg = np.block([[R_pg, t_pg],[0.0, 0.0, 0.0, 1.0]]) # camera2link to plant_frame
                
                T_wp = np.eye(4)
                T_wp[0,3] = centroid[0]
                T_wp[1,3] = centroid[1]
                T_wp[2,3] = centroid[2] # Twp: plant_centroid to world frame
                T_wg = np.matmul(T_wp, T_pg) # camera2_link to world frame
                
                R_wg = T_wg[0:3,0:3]
                q_wg = Rotation.from_matrix(R_wg).as_quat()
                t_wg = T_wg[:,3].tolist() 
                pose_wg = Pose(Point(t_wg[0],t_wg[1],t_wg[2]),
                                Quaternion(q_wg[0],q_wg[1],q_wg[2],q_wg[3]))
                poses_cam_link.append(pose_wg)

                T_g_camframe = self.T_camlink_camframe
                T_w_camframe = np.matmul(T_wg, T_g_camframe)

                R_w_camframe = T_w_camframe[0:3, 0:3]
                q_w_camframe = Rotation.from_matrix(R_w_camframe).as_quat()
                t_w_camframe = T_w_camframe[:,3].tolist()
                pose_w_camframe = Pose(Point(t_w_camframe[0],t_w_camframe[1],t_w_camframe[2]),
                                Quaternion(q_w_camframe[0],q_w_camframe[1],q_w_camframe[2],q_w_camframe[3]))
                
                poses_camframe.append(pose_w_camframe)
                

        return poses_cam_link, poses_camframe

    def publish_main_topics(self, rgb, depth, pose_w_camlink):
        current_time_stamp = rospy.Time.now()
        
        rgb_img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
        rgb_img_msg.header.stamp = current_time_stamp
        rgb_img_msg.header.frame_id = 'camera2_frame'
        self.pub_rgb_image.publish(rgb_img_msg)

        semantic_img, _, __ = get_semantic_image(rgb, self.add_seg_noise)
        semantic_img_msg = self.bridge.cv2_to_imgmsg(semantic_img, encoding='bgr8')
        semantic_img_msg.header.stamp = current_time_stamp
        semantic_img_msg.header.frame_id = 'camera2_frame'
        self.pub_semantic_image.publish(semantic_img_msg)

        depth_img_msg = self.bridge.cv2_to_imgmsg(depth, encoding='passthrough')
        depth_img_msg.header.stamp = current_time_stamp
        depth_img_msg.header.frame_id = 'camera2_frame'
        self.pub_depth_image.publish(depth_img_msg)

        T_w_camlink = ros_pose_to_SE3(pose_w_camlink)
        T_w_camframe = np.matmul(T_w_camlink, self.T_camlink_camframe)
        pose_w_camframe = SE3_to_ros_pose(T_w_camframe)

        self.publish_tf(pose_w_camframe, current_time_stamp)

    def publish_tf(self, pose_w_camframe, current_time_stamp):

        tf_broadcaster = tf2_ros.TransformBroadcaster()
        transform_msg = geometry_msgs.msg.TransformStamped()
        transform_msg.header.frame_id = "world"
        transform_msg.child_frame_id = "camera2_frame"
        transform_msg.transform.translation = pose_w_camframe.position
        transform_msg.transform.rotation = pose_w_camframe.orientation
        transform_msg.header.stamp = current_time_stamp
        tf_broadcaster.sendTransform(transform_msg)
    

if __name__ == '__main__':
    try:
        node = NBVPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
