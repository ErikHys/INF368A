import math

x = [0.5, 2.1, -1.5, 1.1, 3.3, -0.7]

y = []
for xi in x:
    y.append(math.exp(xi)/sum([math.exp(xj) for xj in x]))
print(y)