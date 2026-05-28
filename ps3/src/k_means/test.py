from matplotlib.image import imread
import matplotlib.pyplot as plt

A = imread('peppers-large.tiff')
plt.imshow(A)
plt.show()