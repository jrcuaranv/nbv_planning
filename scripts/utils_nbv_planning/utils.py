import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, Twist, Point, Quaternion

def compute_xyz_vector(width, height, fx, fy, skip_pixel = 5, max_range = 1):
    
    width = int(width)
    height = int(height)
    # Allocate arrays
    x_index = np.array([list(range(0,width, skip_pixel))*(height//skip_pixel)], dtype = '<f4')
    y_index = np.array([[i]*(width//skip_pixel) for i in range(0,height, skip_pixel)], dtype = '<f4').ravel()
    xy_index = np.vstack((x_index, y_index)).T # x,y
    xyd_vect = np.zeros([xy_index.shape[0], 3], dtype = '<f4') # x,y,depth
    
    K = np.array([[fx, 0, width/2],
                    [0, fy, height/2],
                    [0, 0, 1.0]])
    xyd_vect[:,0:2] = xy_index * max_range
    xyd_vect[:,2:3] = max_range
    XYZ_vect = xyd_vect.dot(np.linalg.inv(K).T)
    XYZ_vect = XYZ_vect.T
    XYZ_vect_hom = np.vstack((XYZ_vect, np.ones((1,XYZ_vect.shape[1]))))
    return XYZ_vect_hom


def ros_pose_to_SE3(rospose):
    # rospose in form Pose(point(), quaternion)
    t = np.array([rospose.position.x, rospose.position.y,
                            rospose.position.z]).reshape(3,1)
    R = Rotation.from_quat([rospose.orientation.x,
                                    rospose.orientation.y,
                                    rospose.orientation.z,
                                    rospose.orientation.w]).as_matrix()
    T = np.block([[R, t],[0.0,0.0,0.0,1.0]])
    return T

def SE3_to_ros_pose(T):
    '''
    T a 4x4 matrix
    '''
    q = Rotation.from_matrix(T[0:3,0:3]).as_quat()
    t = T[0:3,3].squeeze()
    ros_pose = Pose(Point(t[0],t[1],t[2]), Quaternion(q[0],q[1],q[2],q[3])) #x,y,z,qx,qy,qz,qw
    return ros_pose