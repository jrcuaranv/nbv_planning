import numpy as np
from geometry_msgs.msg import Point


def softmax(l):
    return np.exp(l) / np.sum(np.exp(l))

def rgb_to_label(rgb):
    if (rgb[0] == 0 and rgb[1] == 0 and rgb[2]==0 ):
        return 'b'
    if (rgb[0] == 255 and rgb[1] == 0 and rgb[2]==0):
        return 'r'
    if (rgb[0] == 0 and rgb[1] == 255 and rgb[2]==0):
        return 'g'
    if (rgb[0] == 255 and rgb[1] == 255 and rgb[2]==255):
        return 'free'
    if (rgb[0] == 100 and rgb[1] == 100 and rgb[2]==100):
        return 'others'
    if (rgb[0] == 127 and rgb[1] == 127 and rgb[2]==127):
        return 'unknown'    

    return '0'
    
class ViewpointEvaluation:
    def __init__(self, RLE_query_, phi, psi):
        self.phi = phi
        self.psi = psi
        self.RLE_query = RLE_query_

        self.l_k = self.phi * np.ones((5, 5))
        self.l_k[:, 0] = 0
        for i in range(4):
            self.l_k[i, i + 1] = self.psi

    def compute_viewpoint_score(self, T_w_camframe, xyz_vector_hom, centroid, method):
        '''
        Computes the information gain for a single viewpoint
        T_w_camframe: SE3. Transform from camframe to world frame
        xyz_vector_hom: (4,:) Endpoints for ray-casting wrt camera frame
        centroid: [x,y,z] Viewpoint target
        method: str Information gain metric 
        '''
        score = 0
        origin_point = np.array([T_w_camframe[0,3], T_w_camframe[1,3], T_w_camframe[2,3]])
        end_points = np.matmul(T_w_camframe, xyz_vector_hom)[:3, :].T
        # RS: Random sampling
        if method == "RS":
            score = np.random.sample()
        # MI: Mutual information
        if method == 'MI':
            RLE_list = self.get_RLE(origin_point, end_points)
            score = self.compute_pose_score_RLE(RLE_list)
        # UVC: Unknown voxels count
        if method == 'UVC':
            for i in range(end_points.shape[0]):
                end_point = [end_points[i,:]]
                RLE_list = self.get_RLE(origin_point, end_point)
                LE_list = RLE_list[0].le_list
                n_unknown_ri = 0 # number of unknown nodes in ray i
                for LE in LE_list:
                    le = LE.le
                    if le[5] == 127.0:
                        n_unknown_ri += le[0]
                    else: 
                        break
                score += n_unknown_ri
        # UVPC: Unknown voxels with proximity count (see Zaenker, 2021. Viewpoint planning for fruit size..)
        if method == 'UVPC':
            max_dist = 0.1
            for i in range(end_points.shape[0]):
                end_point = [end_points[i,:]]
                RLE_list = self.get_RLE(origin_point, end_point)
                LE_list = RLE_list[0].le_list
                n_unknown_ri = 0 # number of unknown nodes in ray i
                score_ri = 0
                for LE in LE_list:
                    le = LE.le
                    if le[5] == 127.0:
                        le_xyz = [le[8], le[9], le[10]]
                        dist = np.linalg.norm(np.array(centroid) - np.array(le_xyz))
                        if dist < max_dist:
                            n_unknown_ri += le[0]
                            score_ri += le[0]*(0.5 + (max_dist - dist)/max_dist)
                    else: 
                        break
                score += score_ri
        
        # OAE: Occlusion aware entropy (Delmerico et al., 2017)
        if method == 'OAE':
            for i in range(end_points.shape[0]):
                end_point = [end_points[i,:]]
                RLE_list = self.get_RLE(origin_point, end_point)
                LE_list = RLE_list[0].le_list
                score_ri = 0
                free_prob_list = []
                occ_prob_list = []
                
                for LE in LE_list:
                    le = LE.le
                    for i in range(int(le[0])): #le[0] number of nodes with the same logodds values
                        free_prob_list.append(1-le[11])
                        occ_prob_list.append(le[11])
                    
                cum_pv = np.array(free_prob_list).cumprod() # probability of visibility
                
                for p_occ, pv in zip(occ_prob_list, cum_pv):
                    H = -p_occ*np.log2(p_occ) - (1-p_occ)*np.log2(1-p_occ)
                    score_ri += H*pv
                
                score += score_ri
        
        # OSAMCEP: occlusion and semantic aware multi-class entropy with proximity count
        if method == 'OSAMCEP':
            max_dist = 0.1
            for i in range(end_points.shape[0]):
                end_point = [end_points[i,:]]
                RLE_list = self.get_RLE(origin_point, end_point)
                LE_list = RLE_list[0].le_list
                score_ri = 0
                free_prob_list = []
                occ_prob_list = []
                dist_list = []
                semantic_list = []
                entropy = []
                for LE in LE_list:
                    le = LE.le
                    for i in range(int(le[0])): #le[0] number of nodes with the same logodds values
                        free_prob_list.append(1-le[11])
                        # occ_prob_list.append(le[11])
                        le_xyz = [le[8], le[9], le[10]]
                        dist = np.linalg.norm(np.array(centroid) - np.array(le_xyz))
                        dist_list.append(dist)
                        semantic_list.append(rgb_to_label([le[5], le[6], le[7]]))
                        chi = np.array([0, le[1], le[2], le[3], le[4]])
                        pi = softmax(chi) #probabilities
                        entropy.append(-(pi*np.log(pi)).sum())
                cum_pv = np.array(free_prob_list).cumprod() # probability of visibility
                
                for H, pv, dist, sem_color in zip(entropy, cum_pv, dist_list, semantic_list):
                    if (dist < max_dist and (sem_color == 'r' or sem_color == 'unknown')):
                        score_ri += H*pv
                
                score += score_ri
        # AE: Average entropy (presented in Delmerico, 2017)
        if method == 'AE':
            occ_prob_threshold = 0.9
            for i in range(end_points.shape[0]):
                end_point = [end_points[i,:]]
                RLE_list = self.get_RLE(origin_point, end_point)
                LE_list = RLE_list[0].le_list
                score_ri = 0
                occ_prob_list = []
                
                for LE in LE_list:
                    le = LE.le
                    for i in range(int(le[0])): #le[0] number of nodes with the same logodds values
                        occ_prob_list.append(le[11])
                    
                for p_occ in occ_prob_list:
                    if p_occ < occ_prob_threshold: #to consider only voxels before occupied nodes
                        score_ri += -p_occ*np.log2(p_occ) - (1-p_occ)*np.log2(1-p_occ) # entropy
                    else:
                        break
                score += score_ri
        
        return score


    def get_RLE(self, origin_point, end_points):
        """
        Perform raycasting for a single ray. It returns a list of values in Run Length Encoding (RLE)
        See A. Asgharivaskasi and N. Atanasov, “Semantic octree mapping and shannon mutual information computation for robot exploration, 2023.
        Each LE value in RLE list includes:
        LE[0]: Number of nodes sharing the same logodds
        LE[1-4]: semantic logodds organized in descending order
        LE[5-7]: RGB values of the most likely semantic class
        LE[8-10]: Coordinates of final node sharing the same logodds
        LE[11]: occupancy probability
        
        """
        origin_point_ros = Point()
        origin_point_ros.x = origin_point[0]
        origin_point_ros.y = origin_point[1]
        origin_point_ros.z = origin_point[2]
        
        end_points_list_ros = []
        
        for end_point in end_points:
            end_point_ros = Point()
            end_point_ros.x = end_point[0]
            end_point_ros.y = end_point[1]
            end_point_ros.z = end_point[2]
            end_points_list_ros.append(end_point_ros)
        
        try:
            resp = self.RLE_query(end_points_list_ros, origin_point_ros)
            return resp.RLE_list
        except:
            print("RLE Service call failed")

    def compute_pose_score_RLE(self, RLE_list):
        """
        Computes mutual information according to paper:
        A. Asgharivaskasi and N. Atanasov, “Semantic octree mapping and shannon mutual information computation for robot exploration, 2023.
        Some changes are made to encourage this metric to focus on specific semantics (fruit)
        """
        pose_score = 0
        for RLE in RLE_list:
            pose_score += self.compute_ray_score(RLE)
        list_aux =[]
        for RLE in RLE_list:
            for LE in RLE.le_list:
                if len(LE.le)>5:
                    list_aux.append(LE.le)
        
        return pose_score

    def compute_ray_score(self, RLE):
        le_list = RLE.le_list
        
        ray_score = 0
        a_0 = 1
        b_0 = 0
        offset_irrelevant = 5.0
        offset_relevant = 2.0
        # offset_free = -2.0
        for LE in le_list:
            le = LE.le
            w = le[0]
            
            
            color = rgb_to_label([le[5],le[6],le[7]])
            # if (color=='free'):
            #     chi = np.array([0, le[1] + offset_free, le[2] + offset_free, le[3] + offset_free, le[4] + offset_free])
            if (color == 'g' or color == 'b' or color == 'others'):
                chi = np.array([0, le[1]+offset_irrelevant, le[2]+offset_irrelevant, le[3]+offset_irrelevant, le[4]+offset_irrelevant])
            if (color=='r'):
                chi = np.array([0, le[1]+offset_relevant, le[2]+offset_relevant, le[3]+offset_relevant, le[4]+offset_relevant])
            if (color=='unknown'):
                chi = np.array([0, le[1], le[2], le[3], le[4]])

            pi = softmax(chi)
            pi_w_q = (pi[0]) ** w
            f = self.compute_f(chi)
            
            a = a_0 * pi[1:]
            a_0 *= pi_w_q
            
            b = b_0 + f[:4]
            b_0 += w * f[4]
            
            frac_1 = (1 - pi_w_q) / (1 - pi[0])
            frac_2 = ((w - 1) * pi_w_q * pi[0] - w * pi_w_q + pi[0]) / ((1 - pi[0]) ** 2)
            
            ray_score += np.sum(a * (b * frac_1 + f[4] * frac_2))

        return ray_score
            
    def compute_f(self, chi):
        chi = np.resize(chi, (1, 5))
        chi_rep = np.repeat(chi, 5, axis=0)
        l_tilde = chi_rep + self.l_k
        exp_term = np.exp(l_tilde)
        # original implementation seems to be wrong. It was updated according to the author's paper
        # big_softmax = exp_term / np.sum(exp_term, axis=1) # wrong broadcasting error
        # f = np.log(np.sum(np.exp(chi)) * big_softmax[:, 0]) + \
        #     np.sum(big_softmax[:, 1:] * self.l_k[:, 1:], axis=1)
        big_softmax = exp_term / np.sum(exp_term, axis=1).reshape((5,1)) #broadcasting error solved
        f = np.log(np.sum(np.exp(chi))/np.sum(exp_term, axis=1)) + np.sum(big_softmax * self.l_k, axis=1)

        return f
    
    def compute_entropy(self, centroids, map_res=0.01, bbox_size =0.1):
        max_dist = bbox_size/2
        entropy = 0
        
        for centroid in centroids:
            # bounding box limits to do raytracing
            xmin = centroid[0] - bbox_size/2
            xmax = centroid[0] + bbox_size/2
            ymin = centroid[1] - bbox_size/2
            ymax = centroid[1] + bbox_size/2
            zmin = centroid[2] - bbox_size/2
            zmax = centroid[2] + bbox_size/2
            for x in np.arange(xmin, xmax, map_res).tolist():
                for y in np.arange(ymin, ymax, map_res).tolist():
                    origin_point = np.array([x, y, zmin])
                    end_point = [np.array([x, y, zmax])]
                    RLE_list = self.get_RLE(origin_point, end_point)
                    LE_list = RLE_list[0].le_list
                    for LE in LE_list:
                        le = LE.le
                        n = le[0]
                        le_xyz = [le[8], le[9], le[10]]
                        dist = np.linalg.norm(np.array(centroid) - np.array(le_xyz))
                        color = rgb_to_label([le[5], le[6], le[7]])
                        if (dist < max_dist and color != 'free'):
                            chi = np.array([0, le[1], le[2], le[3], le[4]])
                            pi = softmax(chi) #probabilities
                            entropy += -n*((pi*np.log(pi)).sum())

            
        return entropy




