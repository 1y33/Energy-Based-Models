import numpy as np
import matplotlib.pyplot as plt

def energy_function(x, y):
    return x**2 + y**2

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = energy_function(X, Y)


plt.contour(X, Y, Z, levels=20)
plt.title("Energy Field (Contour Plot)")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Energy")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Energy Field (3D Surface)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Energy')
plt.show()

plt.imshow(Z, extent=(-5, 5, -5, 5), origin='lower', cmap='hot')
plt.colorbar(label="Energy")
plt.title("Energy Field (Heatmap)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
