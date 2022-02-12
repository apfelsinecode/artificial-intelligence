import data_generator
import numpy as np
import matplotlib.pyplot as plt
import itertools


def main():
    def f(x: float):
        return 2 * x + 1
    domain = (0, 20)

    fig, ax = plt.subplots()
    points = itertools.islice(
        data_generator.generate_points(f, domain_start=domain[0], domain_end=domain[1]),
        20
    )
    # ax.plot([p[0] for p in points], [p[1] for p in points])
    for x, y in points:
        print(x, y)
        ax.plot(x, y)
    plt.savefig("p.png")
    # plt.show()


def loss():
    pass


if __name__ == '__main__':
    main()
