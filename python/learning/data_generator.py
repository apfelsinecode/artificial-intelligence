from __future__ import annotations

import random


def generate_points(function, spread: float = 0.1, domain_start: float = -10, domain_end: float = 10):
    while True:
        x = random.uniform(domain_start, domain_end)
        y = function(x)
        noise = random.uniform(1 - spread, 1 + spread)
        y *= noise
        yield x, y


if __name__ == "__main__":
    f = lambda x: 5
    for _, c in zip(range(10), generate_points(f)):
        # input("enter")

        print(c)
