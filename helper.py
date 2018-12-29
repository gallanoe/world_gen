import numpy as np
from scipy.spatial import ConvexHull, SphericalVoronoi

def normalize(p, r):
    p /= np.linalg.norm(p, axis=0)
    return p * r

def find_centroid(v):
    return np.mean(v, axis=0)

def relax(points, radius, center, num_iter=1):
    relaxed, sv = points, SphericalVoronoi(points, radius, center)
    while num_iter > 0:
        relaxed = np.array([normalize(find_centroid(sv.vertices[region]), radius) for region in sv.regions])
        sv = SphericalVoronoi(relaxed, radius, center)
        num_iter -= 1
    return sv.points, sv