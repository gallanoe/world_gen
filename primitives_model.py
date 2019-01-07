# import libraries
import math
import random
import copy

import helper

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import collections, colors
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, SphericalVoronoi
from collections import deque, namedtuple

# world object 
class World(object):
    
    # defaults
    plate_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
        '#ffffff', '#3a3a3a'
    ]

    def pick_plate_color(self):
        if len(self.plate_colors) > 0:
            color = random.choice(self.plate_colors)
            self.plate_colors.remove(color)
            return color

    class Node(object):
        __slots__ = ('label', 'point', 'neighbors', 'is_rogue', 'region_vertices', 'movement')

        def __init__ (self, label):
            self.label = label
            self.point = np.zeros(3)
            self.neighbors = []
            self.is_rogue = True
            self.region_vertices = []
            self.movement = np.zeros(3)
        
        def has_rogue_neighbor(self):
            for n in self.neighbors:
                if n.is_rogue:
                    return True
            return False
        
        def __repr__(self):
            return  "{label: " + str(self.label) + \
                    ", neighbors: " + str([node.label for node in self.neighbors]) + \
                    ", rogue: " + str(self.is_rogue) + "}"

    class Plate(object):
        __slots__ = ('label', 'start', 'is_growable', 'nodes', 'centroid')

        def __init__(self, label):
            self.label = label
            self.start = None
            self.is_growable = True
            self.nodes = set()
            self.centroid = np.zeros(3)
        
        def update_growable(self):
            self.is_growable = False
            for node in self.nodes:
                if node.has_rogue_neighbor():
                    self.is_growable = True
        
        def __repr__(self):
            return  "{label: " + str(self.label) + \
                    ", growable: " + str(self.is_growable) + \
                    ", nodes: " + str([n.label for n in self.nodes]) + "}"

    def __init__(
            self, npoints=200, radius=4000,
            center=np.zeros(3), nplates=10,
            degree_of_relaxation=5
        ):
        # world attributes
        self.npoints = npoints
        self.radius = radius
        self.degree_of_relaxation = degree_of_relaxation
        self.center = center
        self.nplates = nplates

        # world data
        self.nodes = []
        self.plates = []

    def generate(self):
        self.__generate_nodes()
        # self.__generate_plates()

    def __generate_nodes(self):
        # generate points
        points = np.random.randn(3, self.npoints)
        points = helper.normalize(points, self.radius)
        
        # resort points
        points = np.array(list(zip(points[0], points[1], points[2])))
    
        # relax points
        points, sv = helper.relax(points, self.radius, self.center, self.degree_of_relaxation)
        sv.sort_vertices_of_regions()

        # compute convex hull
        hull = ConvexHull(points)

        # use siplices to determine neighbors using temporary lists
        primitive_nodes = [set() for i in range(self.npoints)]

        for simplex in hull.simplices:
            primitive_nodes[simplex[0]].update([simplex[1], simplex[2]])
            primitive_nodes[simplex[1]].update([simplex[0], simplex[2]])
            primitive_nodes[simplex[2]].update([simplex[0], simplex[1]])

        # use node class to construct representative data structure
        self.nodes = [World.Node(i)  for i in range(self.npoints)]

        for point, node in zip(points, self.nodes):
            node.point = point

        # fill data accordingly
        for node, primitive_node in zip(self.nodes, primitive_nodes):
            for neighbor in primitive_node:
                node.neighbors.append(self.nodes[neighbor])            

        self.sv = sv

    def __generate_plates(self):
        # TODO: Possibly implement probability distribution weighted by distance from center?
        # Possibly implement breadth first search algo

        # use plates class to construct representative data structure
        self.plates = [World.Plate(i) for i in range(self.nplates)]

        # create list for sampling
        unassigned_nodes = self.nodes[:]

        # for each plate, allocate a random starting location
        for plate in self.plates:

            # sample start node
            start = random.choice(unassigned_nodes)      

            # remove start node from unassigned and set rogue to false
            # add node to plate
            unassigned_nodes.remove(start)
            start.is_rogue = False
            plate.nodes.add(start)
            plate.start = start
        
        # create list of growable plates
        growable_plates = self.plates[:]

        # random fill until no plates can grow further
        while len(growable_plates) > 0:

            # sample plate
            plate = random.choice(growable_plates)

            # create sample list of nodes with valid neighbors
            fill_from = random.choice([node for node in plate.nodes if node.has_rogue_neighbor()])
            added_node = random.choice([node for node in fill_from.neighbors if node.is_rogue])
            
            # remove node from sample list and edit accordingly
            unassigned_nodes.remove(added_node)
            added_node.is_rogue = False
            plate.nodes.add(added_node)

            # update list of growable plates
            for p in growable_plates:
                p.update_growable()
            growable_plates = [p for p in growable_plates if p.is_growable]
        
        # Assign 2-vector to center of tectonic plate and
        # compute change of vector tangent to the surface of the sphere
        for plate in self.plates:

            # grab starting point
            point = plate.start.point

            # generate vector orthonormal to point
            vec = np.random.randn(3)
            vec -= vec.dot(point) * point / np.linalg.norm(point) ** 2
            vec /= np.linalg.norm(vec)
            vec *= (self.radius / 10)

            # set movement to vector
            plate.start.movement = vec
            

        # Assign continental and oceanic status plate or sample base height of plate from normal distribution

    def plot(self):
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        grid_x = np.outer(np.sin(theta) * np.sqrt(self.radius * 1.001), np.cos(phi) * np.sqrt((self.radius * 1.001)))
        grid_y = np.outer(np.sin(theta) * np.sqrt(self.radius * 1.001), np.sin(phi) * np.sqrt((self.radius * 1.001)))
        grid_z = np.outer(np.cos(theta) * np.sqrt(self.radius * 1.001), np.ones_like(phi) * np.sqrt((self.radius * 1.001)))


        corner_points = np.array([
            [self.radius*0.7, self.radius*0.7, self.radius*0.7], [self.radius*0.7, -self.radius*0.7, self.radius*0.7], [-self.radius*0.7, self.radius*0.7, self.radius*0.7], [-self.radius*0.7, -self.radius*0.7, self.radius*0.7],
            [self.radius*0.7, self.radius*0.7, -self.radius*0.7], [self.radius*0.7, -self.radius*0.7, -self.radius*0.7], [-self.radius*0.7, self.radius*0.7, -self.radius*0.7], [-self.radius*0.7, -self.radius*0.7, -self.radius*0.7]
        ])
        fig, ax = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={'projection':'3d', 'aspect':'equal'})

        # hide gridlines and labels
        ax.grid(False)
        plt.axis('off')

        ax.plot_wireframe(grid_x, grid_y, grid_z, alpha=0.0, color='k', rstride=1, cstride=1)
        ax.scatter(corner_points[:,0], corner_points[:,1], corner_points[:,2], alpha=0.0)        

        for plate in self.plates: 
            random_color = self.pick_plate_color()
            indices = [n.label for n in plate.nodes]
            for index in indices:
                region = self.sv.regions[index]
                vertices = [self.sv.vertices[region]]
                print("Vertices:", vertices)
                polygon = Poly3DCollection(vertices, facecolors=random_color, edgecolors='k')
                ax.add_collection3d(polygon)
            point = plate.start.point * 1.01
            vec = plate.start.movement
            ax.scatter(point[0], point[1], point[2], c='r', s=10)
            ax.quiver(point[0], point[1], point[2], vec[0], vec[1], vec[2])


def generate_world(n):
    w = World(npoints=n)
    w.generate()
    w.plot()

def gen500():
    generate_world(500)

def gen1000():
    generate_world(1000)

def gen2000():
    generate_world(2000)

def gen5000():
    generate_world(5000)

if __name__ == "__main__":
    w = World()
    w.generate()
    w.plot()

    # import timeit
    # print("Primitives on 500:", timeit.timeit(gen500, number=1))
    # print("Primitives on 1000:", timeit.timeit(gen1000, number=1))
    # print("Primitives on 2000:", timeit.timeit(gen2000, number=1))
    # print("Primitives on 5000:", timeit.timeit(gen5000, number=1))

    
        