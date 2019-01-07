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

    source : reference to Region object. The region object referred to is the source plate \
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
    __slots__ = ('id', 'source', 'nodes', 'color')
    
    null = None
    plates = set()

    def __init__(self, id=len(plates)):
        """
        Create a plate object.
        """
        # assert that id is int as to allow for indexing into adjacency matrix and regions list
        assert(type(id) is int)
        self.id = id                 
        self.source = Region.null       
        self.nodes = set()
        self.color = helper.pick_plate_color() # TODO: determine default color - use random generator or helper function?
        Plate.plates.add(self)

    def __repr__(self):
        """
        Returns "{ id: self.id, source: self.source, nodes: [self.nodes.id] }
        """
        return  "{ id: " + str(self.id) + \
                ", source: " + str(self.source) + \
                ", nodes: " + str([node.id for node in self.nodes]) + "}"

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
    __slots__ = ('id', 'centroid', 'region_vertices', 'neighbors', 'plate', 'movement')
    
    null = None
    regions = set()


    def __init__(self, id=len(regions)):
        """
        Creates a region object. All parameters but id are set to default values. Created \
            object is added to set-type class variable Region.regions.
        """
        # assert id is int as to allow indexing into list of plates
        assert(type(id) is int)
        self.id = id
        self.centroid = np.zeros(3)
        self.region_vertices = np.zeros((3,3))
        self.neighbors = set()
        self.plate = Plate.null
        self.movement = np.zeros(3)
        Region.regions.add(self)

    def __repr__(self):
        """
        Returns "{ id: self.id, neighbors: [neighbors.id], plate: self.plate }
        """
        return  "{ id: " + str(self.id) + \
                ", centroid: " + str(self.centroid) + \
                ", neighbors: " + str(self.neighbors) + " }"
                # ", plate: " + str(self.plate.id) + " }"

class World():
    """  
    World object with set parameters and generator

    Attributes
    ----------
    nregions : number of regions composing the world. Default is 200 (number used for testing)

    radius : radius of the world. Default is 4000 (radius of the Earth)
    
    nplates : number of plates composing the world. Default is 10

    degrees_of_relaxation : number of runs of Lloyd's algorithm applied to regions to equalize \
        area distribution of each region Default is 5

    use_bfs : bool value that determines whether the plate generation algorithm \
        uses a BFS successor finder as added node instead of sampling from pool

    regions : a set-type of references to Region objects 

    plates : a set-type of references to Plate objects
    """

        
    def __init__(
            self, nregions=200, radius=4000,
            nplates=10, degrees_of_relaxation=5,
            use_bfs=True
        ):
        """
        Creates a World object with set parameters. Nothing will be generated yet.
        """
        # world attributes
        self.nregions = nregions
        self.radius = radius
        self.center = np.zeros(3)
        self.nplates = nplates
        self.degrees_of_relaxation = degrees_of_relaxation
        self.use_bfs = use_bfs

        # world data
        self.regions = [] 
        # adding an adjacency matrix might cause a bit of data redundancy, wasting space and time
        # needed to compute BFS however, and might be useful in later portions of the generation algorithm
        self.region_structure = {}
        self.plates = []

    def generate(self):
        """
        Runs generation using generation settings determined by World attributes. \
        Will override previously generated data.
        """
        self.__generate_regions()
        self.__generate_plates()
    
    def __generate_regions(self):
        """
        Algorithm for generating regions.
        """

        # randomly generate points for regions
        points = np.random.randn(3, self.nregions)
        points = helper.normalize(points, self.radius)

        # resort points
        points = np.array(list(zip(points[0], points[1], points[2])))

        # relax points and compute spherical voronoi
        points, sv = helper.relax(points, self.radius, self.center, self.degrees_of_relaxation)
        sv.sort_vertices_of_regions()

        # compute convex hull
        hull = ConvexHull(points)

        # use simplices to determine neighbors and store using temporary data structure
        primitive_regions = [[] for i in range(self.nregions)]
        for simplex in hull.simplices:
            primitive_regions[simplex[0]].extend([simplex[1], simplex[2]])
            primitive_regions[simplex[1]].extend([simplex[0], simplex[2]])
            primitive_regions[simplex[2]].extend([simplex[0], simplex[1]])
        
        # use region class to construct representative data structure
        self.regions = [Region(i) for i in range(self.nregions)]
        for point, region in zip(points, self.regions):
            region.centroid = point
        
        # fill neighbor data accordingly
        for region, primitive_region in zip(self.regions, primitive_regions):
            region.neighbors.update(primitive_region)

        # fill matrix data
        # as the graph is undirected, we will be using a hash-table (dict) 
        # where the keys are unique pairs of regions (nodes)

        for u, v in combinations(list(range(self.nregions)), 2):
            self.region_structure[(u,v)] = 0
            if v in self.regions[u].neighbors:
                self.region_structure[(u,v)] = 1

        # fill region vertices data
        for region, sv_region in zip(self.regions, sv.regions):
            region.region_vertices = np.array([sv.vertices[index] for index in sv_region])

        del sv
        del hull
  
    def __generate_plates(self):
        
        # use Plate class to construct representative data structure
        self.plates = [Plate(i) for i in range(self.nplates)]

        # create duplicate list of regions of sampling
        # keep in mind that list + random.choice is faster than set + random.sample
        unassigned_regions = self.regions[:]

        # for each plate allocate a random starting location
        for plate in self.plates:

            # sample starting node
            source = random.choice(unassigned_regions)

            # remove start node from sample pool
            unassigned_regions.remove(source)

            # set data attributes accordingly
            plate.source = source
            source.plate = plate

            plate.nodes.add(source)

        # create set of growable plates
        growable_plates = self.plates[:]

        # create custom bfs successor finder
        # searched for specific successor that belongs to no plate
        # that succeeds only nodes from the same plate

        # bfs fill algorithm
        if self.use_bfs:

            # construct table of visited nodes - one copy for reuse 
            visited_table = dict([(i, False) for i in range(self.nregions)])

            def find_bfs_successor(plate):

                # assert plate is passed as parameter with valid source
                assert(type(plate) is Plate)
                assert(type(plate.source) is Region)

                # reset visited table
                for v in visited_table:
                    visited_table[v] = False

                # initiate queue for traversal
                queue = [plate.source.id]

                while queue:
                    # grab first element and return if plate value is null
                    s = self.regions[queue.pop(0)]
                    if s.plate is Plate.null:
                        return s
                    # else iterate through neighbors
                    for n in s.neighbors:
                        # retrieve region object
                        r = self.regions[n]
                        # if n's plate value is null, return
                        if r.plate is Plate.null:
                            return r
                        elif r.plate is plate and not visited_table[n]:
                            queue.append(n)             
                            visited_table[n] = True             
                # return the null region if no such successor found
                return Region.null

            # perform random fill using bfs
            while len(growable_plates) > 0 and len(unassigned_regions) > 0:
                
                # sample palte
                plate = random.choice(growable_plates)

                # find node to add using find_bfs_successor
                # if Region.null returned, remove plate from sampling pool
                added_node = find_bfs_successor(plate)
                if added_node is Region.null:
                    growable_plates.remove(plate)
                else:
                    # add node and set data accordingly
                    plate.nodes.add(added_node)
                    unassigned_regions.remove(added_node)
                    added_node.plate = plate

            if len(growable_plates) > 0 or len(unassigned_regions) > 0:
                print("Yeah, you fucked up somewhere.")
                print("len(growable_plates):", len(growable_plates), "len(unassigned_regions):", len(unassigned_regions))
                if len(growable_plates) > 0:
                    print("Still growable plates:")
                    for plate in growable_plates:
                        print(plate)
                if len(unassigned_regions) > 0:
                    print("Unassigned regions:")
                    for region in unassigned_regions:
                        print(region)
                return
        # randomized fill algorithm
        else:
            # perform random fill using bfs
            while len(growable_plates) > 0 and len(unassigned_regions) > 0:
                
                # sample palte
                plate = random.choice(growable_plates)

            
            if len(growable_plates) > 0 or len(unassigned_regions) > 0:
                print("Yeah, you fucked up somewhere.")
                return

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
        plt.axis("off")
        
        ax.scatter(corner_points[:,0], corner_points[:,1], corner_points[:,2], alpha=0.0) 

        for plate in self.plates:
            print(plate.color)       

        for region in self.regions:
            vertices = region.region_vertices
            print(vertices)
            polygon = Poly3DCollection(vertices, facecolors=region.plate.color, edgecolors='k')
            ax.add_collection3d(polygon)

        plt.show()





if __name__ == "__main__":
    w = World(nregions=200)
    w.generate()

        