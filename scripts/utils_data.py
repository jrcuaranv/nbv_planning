import numpy as np

def load_gt_data(model_id):
    """
    Load ground truth pointclouds for ripe fruits only
    Original mesh models downloaded from: https://github.com/Eruvae/ur_with_cam_gazebo/tree/master
    """
    pointcloud_path = '../data/models/capsicum_plant_' + str(model_id)+'/meshes/VG07_'+str(model_id) + '_fruitonly_ripe.xyz'
    
    if (model_id==3):
        xyz_plant_origin = [0,0,0] # in world frame
        # centroids wrt to plant origin
        centroids = [[-0.06993033,  0.06002088,  0.45047608],
                [-0.03700652, -0.04424895,  0.54291276],
                [-0.03933055, -0.02158917,  0.43903008]]
    
    if (model_id==4):
        xyz_plant_origin = [3,0,0]
        centroids = [[-0.08567506, -0.05938319,  0.48737078],
                [-0.00221522,  0.05939227, 0.37176467],
                [-0.1353272,   0.05924012,  0.37919344]]
    if (model_id==5):
        xyz_plant_origin = [6,0,0]
        centroids = [[-1.93412792e-01,  2.69271750e-02,  6.01567992e-01],
                [ 4.48817500e-02,  1.22202833e-01,  7.07265794e-01],
                [-1.13769044e-01,  2.53497944e-02,  7.52088017e-01],
                [-1.69689272e-01, -7.89700389e-02,  6.69736450e-01],
                [ 1.70755556e-04,  7.48190278e-02,  6.59180506e-01]]
    if (model_id==6):
        xyz_plant_origin = [9,0,0]
        centroids = [[-0.13216007, -0.00368707,  0.28290744],
                [-0.01431401,  0.16611881, 0.46723348],
                [-0.08135753, -0.12183831,  0.48662173],
                [-0.14658656,  0.09308882,  0.22029854],
                [-0.08750977,  0.11758169,  0.3193785 ],
                [-0.02030869, -0.01851992,  0.37241832],
                [-0.12469893, -0.00204074,  0.37855879]]
    if (model_id==7):
        xyz_plant_origin = [12,0,0]
        centroids = [[-0.11829382, -0.12515773,  0.41288378],
                [-0.02624446,  0.15150982,  0.30861333],
                [ 0.0116813,  -0.01833606,  0.37178144],
                [-0.09811491,  0.15312738,  0.31624773],
                [-0.125,   0.04,  0.30116626],
                [-0.14491327,  0.12493199,  0.22055095],
                [-0.14072446, -0.02998647,  0.36957208]]

    if (model_id==8):
        xyz_plant_origin = [15,0,0]
        centroids = [[-0.16819473,  0.12251778,  0.38832089],
                [ 0.03517532,  0.15155287,  0.38219086],
                [-0.02515042,  0.12204709,  0.36009117]]

    w_centroids = np.array(centroids) + np.array(xyz_plant_origin) # centroids wrt world frame
    w_centroids = w_centroids.tolist()

    return w_centroids, xyz_plant_origin, pointcloud_path