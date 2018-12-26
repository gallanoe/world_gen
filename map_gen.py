# import neccesities
import math
import random
import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections, colors
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, SphericalVoronoi

# set constants
NPOINTS = 1000
RADIUS = 4000
CENTER = np.array([0, 0, 0])
NPLATES = 5

# generate gridlines
phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta) * np.sqrt(RADIUS), np.cos(phi) * np.sqrt(RADIUS))
y = np.outer(np.sin(theta) * np.sqrt(RADIUS), np.sin(phi) * np.sqrt(RADIUS))
z = np.outer(np.cos(theta) * np.sqrt(RADIUS), np.ones_like(phi) * np.sqrt(RADIUS))

# normalization function
def normalize(points, radius):
    points /= np.linalg.norm(points, axis=0)
    return points * radius

# generate points function
def test_1():
    # generate NPOINTS random points and normalize
    points = np.random.randn(3, NPOINTS)
    points = normalize(points, RADIUS)
    
    # resort points
    points = np.array(list(zip(points[0], points[1], points[2])))
    
    # find_centroid function
    def find_centroid(vertices):
        x = np.mean([vertex[0] for vertex in vertices])
        y = np.mean([vertex[1] for vertex in vertices])
        z = np.mean([vertex[2] for vertex in vertices])
        return [x, y, z]

    # perform voronoi iteration
    def relaxation(points, num_iter):
        relaxed, sv = points, SphericalVoronoi(points, RADIUS, CENTER)
        while num_iter > 0:
            relaxed = np.array([normalize(find_centroid(sv.vertices[region]), RADIUS) for region in sv.regions])
            sv = SphericalVoronoi(relaxed, RADIUS, CENTER)
            num_iter -= 1
        return sv.points, sv

    # run relaxation 20 times
    points, sv = relaxation(points, 20)
    sv.sort_vertices_of_regions()

    # compute convex hull of spherical points
    hull = ConvexHull(points)

    # combine convex hull and spherical voronoi data into combined class

    # grab simplices from convex hull
    simplices = hull.simplices[:]

    # we use lists (implemented as arrays) as the information is static
    # each element is a list containing indices of neighbors
    nodes_primitives = [set() for i in range(NPOINTS)]
    for simplex in simplices:
        nodes_primitives[simplex[0]].update([simplex[1], simplex[2]])
        nodes_primitives[simplex[1]].update([simplex[0], simplex[2]])
        nodes_primitives[simplex[2]].update([simplex[0], simplex[1]])

    # to increase performance of random fill, create data structure
    # combine both convex hull and spherical voronoi
    class Node(object):
        __slots__ = ('label', 'neighbors', 'rogue')

        def __init__(self, label):
            self.label = label
            self.neighbors = []
            self.rogue = True
        
        def __repr__(self):
            # return str(self.label)
            return "{label: " + str(self.label) + ", neighbors: " + str([node.label for node in self.neighbors]) + ", rogue: " + str(self.rogue) + "}"

        def has_rogue_neighbor(self):
            for n in self.neighbors:
                if n.rogue:
                    return True
            return False

    # create empty node classes for each node
    nodes = [Node(i) for i in range(NPOINTS)]

    # fill data accordingly
    for node, node_primitive in zip(nodes, nodes_primitives):
        for neighbor in node_primitive:
            node.neighbors.append(nodes[neighbor])
    

    # create plates data structure
    class Plate(object):
        __slots__ = ('label', 'growable', 'nodes')

        def __init__(self, label):
            self.label = label
            self.growable = True
            self.nodes = set()
        
        def __repr__(self):
            return "{label: " + str(self.label) + ", growable: " + str(self.growable) + ", nodes: " + str([n.label for n in self.nodes]) + "}"
        
        def is_growable(self):
            return len([node for node in self.nodes if node.has_rogue_neighbor()]) > 0

    plates = [Plate(i) for i in range(NPLATES)]
    usable_plates = plates
    unassigned_nodes = nodes

    # allocate a random starting location
    for plate in plates:

        # sample start node
        start = random.sample(unassigned_nodes,k=1)[0]

        # remove start node from unassigned
        unassigned_nodes.remove(start)

        # set rogue to false
        start.rogue = False

        # add start node to plate
        plate.nodes.add(start)

    # perform random fill
    iter_number = 0 
    while len(usable_plates) > 0:
        # print('####################################################################')
        # print("Usable plates:", [p.label for p in usable_plates])
        # print("Usable nodes:", [n.label for n in unassigned_nodes])
        # print("Iteration:", iter_number)
        
        # select growable plate
        plate = random.choice(usable_plates)
        # print("Chosen plate:", plate)
        
        # create temp list with nodes with valid neighbors
        usable_fill_froms = [node for node in plate.nodes if node.has_rogue_neighbor()]
        # print("Usable fill froms:", [node.label for node in usable_fill_froms])

        # select random node from temp list and select neighbor from node
        fill_from = random.sample(usable_fill_froms,k=1)[0]
        # print("Fill from:", fill_from)

        # create temp list with rogue nodes
        usable_nodes = [node for node in fill_from.neighbors if node.rogue]
        added_node = random.sample(usable_nodes,k=1)[0]
        # print("Added node:", added_node)
        # print("Added node neighbors:")
        # for neighbor in added_node.neighbors:
        #     print(neighbor, added_node.label in [n.label for n in neighbor.neighbors])

        # remove added node from neighbors and unassigned
        unassigned_nodes.remove(added_node)

        # set rogue to false
        added_node.rogue = False
        
        # add added node to plate
        plate.nodes.add(added_node)
        
        iter_number += 1

        for p in usable_plates:
            p.growable = p.is_growable()
        usable_plates = [p for p in usable_plates if p.growable]

    plate_num = 0
    fig, ax = plt.subplots(NPLATES, 1, figsize=(25, NPLATES * 25), subplot_kw={'projection':'3d', 'aspect':'equal'})

    for plate in plates:     
        
        ax[plate_num].plot_wireframe(x, y, z, alpha=0.2, color='k', rstride=1, cstride=1)

        random_color = colors.rgb2hex(np.random.rand(3))
        indices = [n.label for n in plate.nodes]
        for i in range(len(indices)):
            point = sv.points[indices[i]]
            region = sv.regions[indices[i]]

            ax[plate_num].scatter(point[0], point[1], point[2], s=50, c=random_color)
            polygon = Poly3DCollection([sv.vertices[region]], alpha=0.7)
            polygon.set_color(random_color)
            ax[plate_num].add_collection3d(polygon)

        plate_num += 1
    
    fig.savefig('foo_1.pdf')

    plt.close()

def test_2():
    # generate NPOINTS random points and normalize
    points = np.random.randn(3, NPOINTS)
    points = normalize(points, RADIUS)
    
    # resort points
    points = np.array(list(zip(points[0], points[1], points[2])))
    
    # find_centroid function
    def find_centroid(vertices):
        x = np.mean([vertex[0] for vertex in vertices])
        y = np.mean([vertex[1] for vertex in vertices])
        z = np.mean([vertex[2] for vertex in vertices])
        return [x, y, z]

    # perform voronoi iteration
    def relaxation(points, num_iter):
        relaxed, sv = points, SphericalVoronoi(points, RADIUS, CENTER)
        while num_iter > 0:
            relaxed = np.array([normalize(find_centroid(sv.vertices[region]), RADIUS) for region in sv.regions])
            sv = SphericalVoronoi(relaxed, RADIUS, CENTER)
            num_iter -= 1
        return sv.points, sv

    # run relaxation 20 times
    points, sv = relaxation(points, 20)
    sv.sort_vertices_of_regions()

    # compute convex hull of spherical points
    hull = ConvexHull(points)

    # combine convex hull and spherical voronoi data into combined class

    # grab simplices from convex hull
    simplices = hull.simplices[:]

    # we use lists (implemented as arrays) as the information is static
    # each element is a list containing indices of neighbors
    nodes_primitives = [set() for i in range(NPOINTS)]
    for simplex in simplices:
        nodes_primitives[simplex[0]].update([simplex[1], simplex[2]])
        nodes_primitives[simplex[1]].update([simplex[0], simplex[2]])
        nodes_primitives[simplex[2]].update([simplex[0], simplex[1]])

    # create plates data structure
    plates = [set() for i in range(NPLATES)]
    unassigned_nodes = set(range(NPOINTS))

    # allocate a random starting location
    for plate in plates:
        start = random.sample(unassigned_nodes,k=1)[0]
        unassigned_nodes.remove(start)
        plate.add(start)

        
    # perform random fill
    while len(unassigned_nodes) > 0:
        added_node = None
        plate = None
        while added_node not in unassigned_nodes:
            plate = random.choice(plates)
            added_node = None
            fill_from = random.sample(plate,k=1)[0]
            added_node = random.sample(nodes_primitives[fill_from],k=1)[0]
        plate.add(added_node)
        unassigned_nodes.remove(added_node)
    
    plate_num = 0
    fig, ax = plt.subplots(NPLATES, 1, figsize=(25, NPLATES * 25), subplot_kw={'projection':'3d', 'aspect':'equal'})

    for plate in plates:
        ax[plate_num].plot_wireframe(x, y, z, alpha=0.2, color='k', rstride=1, cstride=1)
        random_color = colors.rgb2hex(np.random.rand(3))
        indices = list(plate)
        for i in range(len(indices)):
            point = sv.points[indices[i]]
            region = sv.regions[indices[i]]

            ax[plate_num].scatter(point[0], point[1], point[2], s=50, c=random_color)
            polygon = Poly3DCollection([sv.vertices[region]], alpha=0.7)
            polygon.set_color(random_color)
            ax[plate_num].add_collection3d(polygon)

        plate_num += 1

    fig.savefig('foo_2.pdf')

if __name__ == '__main__':
    import timeit
    print(timeit.timeit(test_1, number=1))
    print(timeit.timeit(test_2, number=1))



