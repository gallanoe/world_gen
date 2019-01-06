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
        """
        TODO: RETHINK PLATE ARCHITECTURE
        Ideas:
            (1) seems to be the best option

            1.) add region property that identifies region node belongs to - also create multidimensional array
            according to plates and regions for ease of access (O(1) access) 
                Pros: No data redundancies - ease of access - all node searches are simply dependent on a single Graph object
                Cons: None to think of
            
            NOTES:
                seems like we're going to have to run traversal with each iteration
                luckily, we only run it on the plate selected
        """

        # time to implement architecture
        # add default plate for each node
        for node in self.regions.nodes():
            self.regions.nodes[node]['plate'] = None

        # generate default plates
        self.plates = [{'id': i, 'nodes': set()} for i in range(self.nplates)]

        # generate starting spots
        unassigned_regions = set(range(self.npoints))

        for plate in self.plates:
            start = random.sample(unassigned_regions, 1)[0]
            plate['start'] = start
            plate['nodes'].add(start)
            unassigned_regions.remove(start)
            self.regions.nodes[start]['plate'] = plate['id']
        
        # create traversal for each plate
        # probably temporary code

        # we're gonna have to create our own BFS!!!!!!
        # returns next node in traversal of subgraph determined
        # by union of source's plate and default plate 
        # that has plate value of None
        def __return_bfs_succ(source):
            
            # assert that source can not have plate of None
            assert(self.regions.nodes[source]['plate'] is not None)

            # copy graph as to not edit original graph
            graph = self.regions
            for node in graph.nodes():
                graph.nodes[node]['visited'] = False

            # set source has visited (we are starting from there)
            graph.nodes[source]['visited'] = True

            # get source plate id
            pid = graph.nodes[source]['plate']

            # start queue
            queue = [source]

            # iterate through queue
            while queue:
                # grab first element and return if plate value is None
                s = queue.pop(0)
                if graph.nodes[s]['plate'] is None:
                    return s
                # else iterate through neighbors
                for n in self.regions.neighbors(s):
                    if graph.nodes[n]['plate'] is None:
                        return n
                    elif graph.nodes[n]['plate'] == pid and not graph.nodes[n]['visited']:
                        queue.append(n)
                        graph.nodes[n]['visited'] = True        
            return None
                
        # make copy of list but keep references  
        growable_plates = self.plates[:]

        # random bfs fill
        while len(unassigned_regions) > 0:
            # select random plate
            plate = random.choice(growable_plates)
            
            # grab the next node to add 
            added_node = __return_bfs_succ(plate['start'])
            
            # if added_node is None then plate is not growable 
            # remove plate from pool and select new
            if added_node is None:
                growable_plates.remove(plate)
            else:
                # add node and set associated data accordingly
                plate['nodes'].add(added_node)
                unassigned_regions.remove(added_node)   
                self.regions.nodes[added_node]['plate'] = plate['id']
        

    def generate(self):
        self.__generate_regions()
        self.__generate_plates()

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

        ax.grid(False)
        plt.axis('off')

        ax.scatter(corner_points[:,0], corner_points[:,1], corner_points[:,2], alpha=0.0) 

        for node in self.regions.nodes.data():
            
            # draw vertices

            vertex_indices = node[1]['vertices']
            vertices = [self.region_vertices[vertex_index] for vertex_index in vertex_indices]
            polygon = Poly3DCollection([vertices], facecolors=World.__plate_colors[node[1]['plate']], edgecolors='k')
            ax.add_collection3d(polygon)


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
    import timeit
    print("NetworkX on 500:", timeit.timeit(gen500, number=1))
    print("NetworkX on 1000:", timeit.timeit(gen1000, number=1))
    print("NetworkX on 2000:", timeit.timeit(gen2000, number=1))
    print("NetworkX on 5000:", timeit.timeit(gen5000, number=1))