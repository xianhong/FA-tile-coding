from __future__ import print_function
from tile3 import IHT,tileswrap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

lims = [(0, 2.0 * np.pi)] * 2
def target_ftn(x, y, noise=True):
    return np.sin(x) + np.cos(y) + noise * np.random.randn() * 0.1


maxSize = 1000
iht = IHT(maxSize)
weights = [0]*maxSize
numTilings = 8
stepSize = 0.1/numTilings

def mytiles(x, y):
    scaleFactor = 10.0/(2*np.pi)
    #return tiles(iht, numTilings, list((x*scaleFactor,y*scaleFactor)))
    return tileswrap(iht, numTilings, list((x*scaleFactor,y*scaleFactor))
                     ,[10,10])

def learn(x, y, z):
    tiles = mytiles(x, y)
    estimate = 0
    for tile in tiles: 
        estimate += weights[tile]                  #form estimate
    error = z - estimate
    for tile in tiles: 
        weights[tile] += stepSize * error          #learn weights

def test(x, y):
    tiles = mytiles(x, y)
    estimate = 0
    for tile in tiles: 
        estimate += weights[tile]
    return estimate 

# randomly sample target function until convergence
timer = time.time()
batch_size = 100
for iters in range(100):
    mse = 0.0
    for b in range(batch_size):
      xi = lims[0][0] + np.random.random() * (lims[0][1] - lims[0][0])
      yi = lims[1][0] + np.random.random() * (lims[1][1] - lims[1][0])
      zi = target_ftn(xi, yi)
      learn(xi,yi,zi)
      mse += (test(xi, yi) - zi) ** 2
    mse /= batch_size
    print('samples:', (iters + 1) * batch_size, 'batch_mse:', mse)
print('elapsed time:', time.time() - timer)

# get learned function
print('mapping function...')

res=200
x = np.arange(-np.pi, 3*np.pi, 4*np.pi / res)
y = np.arange(-np.pi, 3*np.pi, 4*np.pi / res)

z = np.zeros([len(x), len(y)])
for i in range(len(x)):
    for j in range(len(y)):
      z[i, j] = test(x[i], y[j])

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
plt.show()

print("Indexed Hash Table count:",str(iht.count()))
