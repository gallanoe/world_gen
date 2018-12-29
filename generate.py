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

class TectonicPlate(nx.Graph):
    """ TectonicPlate object structure, represented by a graph with additional attributes and methods """
    
    def __init__(self, id):
        self.id = id
        self.regions = nx.Graph()



class World(object):
    """  World object with set parameters and generator """

    # defaults
    __plate_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
        '#ffffff', '#3a3a3a'
    ]

    def pick_plate_color(self):
        if len(self.__plate_colors) > 0:
            color = random.choice(self.__plate_colors)
            self.__plate_colors.remove(color)
            return color

    def __init__(
            self, npoints=200, radius=4000,
            center=np.zeros(3), nplates=10,
            degrees_of_relaxation=5
        ):
        # world attributes
        self.npoints = npoints
        self.radius = radius
        self.center = center
        self.nplates = nplates
        self.degrees_of_relaxation = degrees_of_relaxation

        # world data
        self.regions = []
        self.structure = nx.Graph()
        self.plates = [nx.Graph() for i in range(self.nplates)]


    def __generate_regions(self):
        
        # generate points
        points = np.random.randn(self.npoints, 3)
        points = helper.normalize(points, self.radius)

        # relax points
        points, sv = helper.relax(points, self.radius, self.center, self.degrees_of_relaxation)
        sv.sort_vertices_of_regions()

        # compute convex hull
        hull = ConvexHull(points)

        for simplex in hull.simplices:
            

    def generate(self):
        self.__generate_regions()

if __name__ == '__main__':
    w = World()
    w.generate()
 