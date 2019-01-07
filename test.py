import timeit

import matplotlib.pyplot as plt
import numpy as np

import networkx_model as nx_model
import primitives_model as pr_model
import model as model


def networkx_generate_world(n):
	w = nx_model.World(npoints=n)
	w.generate()
	w.plot()

def networkx_test():
    def networkx_gen500():
        networkx_generate_world(500)

    def networkx_gen1000():
        networkx_generate_world(1000)

    def networkx_gen2000():
        networkx_generate_world(2000)

    def networkx_gen5000():
        networkx_generate_world(5000)

    print("NetworkX on 500:", timeit.timeit(networkx_gen500, number=1))
    print("NetworkX on 1000:", timeit.timeit(networkx_gen1000, number=1))
    print("NetworkX on 2000:", timeit.timeit(networkx_gen2000, number=1))
    print("NetworkX on 5000:", timeit.timeit(networkx_gen5000, number=1))
    
def primitives_generate_world(n):
        w = pr_model.World(npoints=n)
        w.generate()

def primitives_test():
    def primitives_gen500():
        primitives_generate_world(500)

    def primitives_gen1000():
        primitives_generate_world(1000)

    def primitives_gen2000():
        primitives_generate_world(2000)

    def primitives_gen5000():
        primitives_generate_world(5000)

    print("Primitives on 500:", timeit.timeit(primitives_gen500, number=1))
    print("Primitives on 1000:", timeit.timeit(primitives_gen1000, number=1))
    print("Primitives on 2000:", timeit.timeit(primitives_gen2000, number=1))
    print("Primitives on 5000:", timeit.timeit(primitives_gen5000, number=1))

def model_generate_world(n):
    w = model.World(nregions=n)
    w.generate()

def model_test():
    def model_gen500():
        primitives_generate_world(500)

    def model_gen1000():
        primitives_generate_world(1000)

    def model_gen2000():
        primitives_generate_world(2000)

    def model_gen5000():
        primitives_generate_world(5000)

    print("Model on 500:", timeit.timeit(model_gen500, number=1))
    print("Model on 1000:", timeit.timeit(model_gen1000, number=1))
    print("Model on 2000:", timeit.timeit(model_gen2000, number=1))
    print("Model on 5000:", timeit.timeit(model_gen5000, number=1))

def skip_func(functions):
    for func in functions:
        try:
            func()
        except KeyboardInterrupt:
            print()
            print("SKIPPING")
            print()

import random

def test_choice(a):
    assert(type(a) is list)
    choice = random.choice(a)
    a.remove(choice)
    return choice

def test_sample(a):
    assert(type(a) is set)
    choice = random.sample(a, 1)[0]
    a.remove(choice)
    return choice

if __name__ == "__main__":

    skip_func([primitives_test, model_test])
    

	# time to fully test primitives
	# xx = np.array([100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
	# yy = []
	# for x in xx:
	# 	y = timeit.timeit(stmt="primitives_generate_world({})".format(x), setup="from __main__ import primitives_generate_world", number=1)
	# 	print("Testing primitives", x, ":", y)
	# 	yy.append(y)
	# yy = np.array(yy)
	# print("xx:", xx)
	# print("yy:", yy)

	# fig, ax = plt.subplots(1, 1, figsize=(15, 15))
	# ax.scatter(xx, yy)
	# plt.show()

    # choice_time = timeit.timeit(stmt="test_choice(arr)", setup="from __main__ import test_choice\narr = list(range(100000))", number=100)
    # sample_time = timeit.timeit(stmt="test_sample(arr)", setup="from __main__ import test_sample\narr = set(list(range(100000)))", number=100)
    # print("random.choice:", choice_time)
    # print("random.sample:", sample_time)


    



