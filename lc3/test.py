import numpy as np

x = np.arange(1,50)
y = np.arange(1,100)

xx, yy = np.meshgrid(x, y)

print("xx\n", xx)
print("yy\n", yy)