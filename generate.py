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
from collections import deque, namedtuple, OrderedDict
from itertools import combinations

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
        self.region_vertices = None
        self.regions = nx.Graph()
        self.regions.add_nodes_from(list(range(self.npoints)))

    def __generate_regions(self):
        
        # generate points
        points = np.random.randn(3, self.npoints)
        points = helper.normalize(points, self.radius)

        # resort points
        points = np.array(list(zip(points[0], points[1], points[2])))

        # relax points
        points, sv = helper.relax(points, self.radius, self.center, self.degrees_of_relaxation)
        sv.sort_vertices_of_regions()

        # add points to world data
        for node, point in zip(self.regions.nodes(), points):
            self.regions.nodes[node]['point'] = point

        # compute convex hull
        hull = ConvexHull(points)

        # use networkx graph structure and fill data accordingly
        for simplex in hull.simplices:
            for u, v in combinations([0, 1, 2], 2):
                self.regions.add_edge(simplex[u], simplex[v])

        # check neighbor data
        # for node in self.regions.nodes:
        #     print("N:", node, "Neighbors:", list(self.regions.neighbors(node)))
        
        # copy data from spherical voronoi (before deletion)
        self.region_vertices = sv.vertices[:]
        for node, region in zip(self.regions, sv.regions):
            self.regions.nodes[node]['vertices'] = region[:]

        # check vertex data
        # for point, vertices, data in zip(points, sv.regions, self.regions.nodes.data()):
        #     print("V:", vertices)
        #     print("D:", data)

    def __generate_plates(self):

        # define helper functios relevent to plate generation 
        def __has_rogue_neighbor(self, region):
            for neighbor in self.regions.neighbors(region):
                if self.regions.nodes[neighbor]['is_rogue']:
                    return True
            return False
        
        def __is_growable(self, plate):
            for node in plate.nodes:
                if self.__has_rogue_neighbor(node):
                    return True
            return False

        """
        TODO: RETHINK PLATE ARCHITECTURE
        Ideas:
            (1) seems to be the best option

            1.) add region property that identifies region node belongs to - also create multidimensional array
            according to plates and regions for ease of access (O(1) access) 
                Pros: No data redundancies - ease of access - all node searches are simply dependent on a single Graph object
                Cons: None to think of
            
            2.) 
        """

        # time to implement architecture

        # generate default plates
        self.plates = [{'nodes': set()} for _ in range(self.nplates)]

        # generate starting spots
        unassigned_regions = set(range(self.npoints))

        for plate in self.plates:
            start = random.sample(unassigned_regions, 1)[0]
            plate['start'] = start
            plate['nodes'].add(start)
            unassigned_regions.remove(start)
        
        # create traversal for each plate
        # probably temporary code

        # we're gonna have to create our own BFS!!!!!!
        def bfs_search(graph, source):

            graph_copy = copy.deepcopy(graph)
            for node in graph_copy.nodes:
                graph_copy.nodes[node]['visited'] = False
            graph_copy.nodes[source]['visited'] = True

            queue = [source]
            traversal = []

            while queue:
                s = queue.pop(0)
                traversal.append(s)
                neighbors = graph_copy.neighbors(s)
                for n in neighbors:
                    if not graph_copy.nodes[n]['visited']:
                        queue.append(n)
                        graph_copy.nodes[n]['visited'] = True
            
            return traversal

        self.bfs_search = bfs_search
        for plate in self.plates:

            # generate reverse bfs
            reverse_bfs = [(node, None) for node in bfs_search(self.regions, plate['start'])][1:]
            # construct ordered dict with key as entry - O(1) pop
            # using ordered dict as ordered set
            plate['bfs'] = OrderedDict(reverse_bfs)

        # remove starts of other plates from other bfs'
        for plate in self.plates:
            for other_plate in self.plates:
                if other_plate['start'] in plate['bfs']:
                    del plate['bfs'][other_plate['start']]
                
        # make copy of list but keep references  
        growable_plates = self.plates[:]

        # random bfs fill
        while len(unassigned_regions) > 0:
            # select random plate
            plate = random.choice(growable_plates)
            # print(plate)

            # select node based on bfs
            added_node = plate['bfs'].popitem()[0]
            plate['nodes'].add(added_node)

            # remove node from other bfs' to ensure they are not added again
            for other_plate in self.plates:
                try:
                    del other_plate['bfs'][added_node]
                except:
                    pass

                # remove plates from growable if no more plates to add
                if len(other_plate['bfs']) <= 0 and other_plate in growable_plates:
                    growable_plates.remove(other_plate)
            
            # remove added node from unassigned nodes
            unassigned_regions.remove(added_node)   




        # for plate in self.plates:
        #     print(plate)
                
        

        
        
        
            

        
        
                


    def generate(self):
        self.__generate_regions()
        self.__generate_plates()



if __name__ == '__main__':
    w = World()
    w.generate()
 