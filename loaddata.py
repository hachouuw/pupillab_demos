import pickle
from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np


# open a file, where you stored the pickled data
file = open('mouse_xy.pkl', 'rb')

# x_dim: 3840, y_dim: 1200

# dump information to that file
data = pickle.load(file)

# close the file
file.close()


x = [i[0] for i in np.array(data[2:])]
y = [i[1] for i in np.array(data[2:])]


plt.scatter(x, y, c ="blue")
plt.xlim
plt.show()
