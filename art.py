import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.axis('off')

Y, X = np.mgrid[-4:4:0.05, -4:4:0.05]
U = np.sin(Y) - np.cos(X)
V = np.cos(Y) + np.sin(X)
plt.streamplot(X, Y, U, V, density=2, color='black', linewidth=0.4)

plt.savefig("rug_vector_field.svg", bbox_inches='tight')
plt.show()
