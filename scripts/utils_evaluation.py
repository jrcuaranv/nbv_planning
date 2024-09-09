import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load your data from the .txt file

def compute_surface_coverage(data_gt, data_map, offset, dist_threshold):
    nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data_map)
    surface_points_count = 0
    for point in data_gt:
        w_point = point + np.array(offset)
        distance, index = nearest_neighbors.kneighbors(w_point.reshape(1,3))
        if distance <= dist_threshold:
            surface_points_count +=1
    surface_coverage = surface_points_count / data_gt.shape[0]
    print("Surface coverage:", surface_coverage*100, "%")
    return surface_coverage