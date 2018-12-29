import math
import random
import copy

import helper

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import collections, colors
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, SphericalVoronoi
from collections import deque, namedtuple

radius = 4000
npoints = 10

# create point
phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
grid_x = np.outer(np.sin(theta) * np.sqrt(radius * 1.001), np.cos(phi) * np.sqrt((radius * 1.001)))
grid_y = np.outer(np.sin(theta) * np.sqrt(radius * 1.001), np.sin(phi) * np.sqrt((radius * 1.001)))
grid_z = np.outer(np.cos(theta) * np.sqrt(radius * 1.001), np.ones_like(phi) * np.sqrt((radius * 1.001)))

# generate random point
points = np.random.randn(3, npoints)

points /= np.linalg.norm(points, axis=0)
points *= radius

points = np.array(list(zip(points[0], points[1], points[2])))

def generate_vecs(points):
    vecs = []
    for point in points:
        # treat point as vector (vectors from radius are othonormal to the surface of sphere)
        x = np.random.randn(3)
        x -= x.dot(point) * point / np.linalg.norm(point) ** 2
        x /= np.linalg.norm(x)
        x *= (radius / 50)
        vecs.append(x)
    return np.array(vecs)
        
vecs = generate_vecs(points)
print(vecs)
    

fig, ax = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={'projection':'3d', 'aspect':'equal'})

# hide gridlines and labels
ax.grid(False)
plt.axis('off')

ax.plot_wireframe(grid_x, grid_y, grid_z, alpha=0.2, color='k', rstride=1, cstride=1)
ax.scatter(points[:,0], points[:,1], points[:,2], s=10, c='r')
ax.quiver(points[:,0], points[:,1], points[:,2], vecs[:,0], vecs[:,1], vecs[:,2], length=3)

for angle in range(360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)