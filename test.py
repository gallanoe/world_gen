import networkx_model as nx_model
import primitives_model as pr_model
import timeit

def networkx_test():
    def generate_world(n):
        w = nx_model.World(npoints=n)
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

    print("NetworkX on 500:", timeit.timeit(gen500, number=1))
    print("NetworkX on 1000:", timeit.timeit(gen1000, number=1))
    print("NetworkX on 2000:", timeit.timeit(gen2000, number=1))
    print("NetworkX on 5000:", timeit.timeit(gen5000, number=1))
    
def primitives_test():
    def generate_world(n):
        w = pr_model.World(npoints=n)
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

    print("Primitives on 500:", timeit.timeit(gen500, number=1))
    print("Primitives on 1000:", timeit.timeit(gen1000, number=1))
    print("Primitives on 2000:", timeit.timeit(gen2000, number=1))
    print("Primitives on 5000:", timeit.timeit(gen5000, number=1))

def skip_func(functions):
    for func in functions:
        try:
            func()
        except KeyboardInterrupt:
            print()
            print("SKIPPING")
            print()

if __name__ == "__main__":
	skip_func([networkx_test, primitives_test])