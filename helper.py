import numpy as np
import random
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


#######################################
# keeping old code for reference later
#######################################
__plate_colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
    '#ffffff', '#3a3a3a'
]

def pick_plate_color():
    """ 
    Simple function for picking unique and distinct color for plates. 
    Will exhaust after number of plate colors calls.
    """ 
    if len(__plate_colors) > 0:
        color = random.choice(__plate_colors)
        __plate_colors.remove(color)
        return color

#######################################
# keeping old code for reference later
#######################################    