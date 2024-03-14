import matplotlib.pyplot as plt
import numpy as np
# y = np.arange(0, 5, 1)
# x = np.array([a * a for a in y])

y_2 = [0, 1, 2, 3]
x_2 = [i+1 for i in y_2]

# plt.plot(x, y, "r--o", x_2, y_2, "b:v")
# marker o, b, ^, <, >, 2, 3, 4, # s, p, *, H, h, +, x, D, d, |, -

plt.grid()
plt.show()

x = np.arange(-2*np.pi, 2*np.pi, 0.1)
y = np.cos(x)

plt.grid()
plt.plot(x, y)
plt.fill_between(x, y)
plt.show()
