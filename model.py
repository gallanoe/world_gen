# import libraries
import copy
import math
import random
from collections import OrderedDict, deque, namedtuple
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import collections, colors
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, SphericalVoronoi

import helper

class Plate:
    """
    Represents a plate, a collection of neighboring regions.

    Attributes
    ----------
    id : the label used to identify the plate

    start : reference to Region object. The region object referred to is the source plate \
        used for breadth-first search of region structure when performing probabilistic fill.
    
    nodes : a set of references to Region objects. \
        A set is used for the constant-time look-up, insertion, and deletion. \
        Also checks for uniqueness as to ensure that no duplicate regions are inserted.

    color : color used to represent plate in render/graph.
    
    Class Attributes
    ----------------
    Plate.null : the nullified Plate object. Used as a placeholder for variables \
        requiring a reference to Plate object but yet assigned a value.

    Plate.plates : a set of references to Plate objects that keeps track of all \
        created Plate objects
    """
    __slots__ = ('id', 'start', 'nodes')
    
    null = None
    plates = set()

    def __init__(self, id=len(Plate.plates)):
        """
        Create a plate object.
        """
        self.id = id                 
        self.start = Region.null       
        self.nodes = set()
        self.color = '#e6194b' # TODO: determine default color - use random generator or helper function?

    

class Region:
    """
    Represents a region, analogous to a node in the region structure of the model world \
        where each edge of the graph represents a neighboring relationship.

    Attributes
    ----------
    id : the label used to identify the region

    centroid : a numpy array of shape (3) representing the position of the centroid of the region

    region_vertex_indices : a numpy array of shape (3, n) where n represents the number of vertices that \
        compose the region

    neighbors : a set of references to other Region objects that neighbor self

    plate : a reference to a Plate object in which the region belongs to

    movement : a numpy array of shape (3) representing a vector determining the direction \
        in which the region was moving. TODO: might consider using a vector of shape (2) \
        mapped to the surface of a sphere using a Jacobian.

    Class Attributes
    ----------------
    Region.null : the nullified Region object. Used as a placeholder for variables \
        requiring a reference to Region object but yet assigned a value.

    Region.regions : a set of references to Region objects that keeps track of all \
        created Region objects
    """
    __slots__ = ('id', 'centroid', 'region_vertex_indices', 'neighbors', 'plate', 'movement')
    
    null = None
    regions = set()


    def __init__(self, id=len(Region.regions)):
        """
        Creates a region object. All parameters but id are set to default values. Created \
            object is added to set-type class variable Region.regions.
        """
        self.id = id
        self.centroid = np.zeros(3)
        self.region_vertices = np.zeros((3,3))
        self.neighbors = set()
        self.plate = Plate.null
        self.movement = np.zeros(3)

    def __repr__(self):
        """
        Returns "{ id: self.id, neighbors: [neighbors.id], plate: self.plate }
        """
        return  "{ id: " + str(self.id) + \
                ", neighbors: " + str([neighbor.id for neighbor in self.neighbors]) + \
                ", plate: " + str(self.plate.id) + " }"

class World():
    """  
    World object with set parameters and generator
    """

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

    def pick_plate_color(self):
        """ 
        Simple function for picking unique and distinct color for plates. 
        Will exhaust after number of plate colors calls.
        """ 
        if len(self.__plate_colors) > 0:
            color = random.choice(self.__plate_colors)
            self.__plate_colors.remove(color)
            return color

    #######################################
    # keeping old code for reference later
    #######################################    
        
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
        self.region_vertices = None
        self.regions = 