import numpy as np
import matplotlib.pyplot as plt

# ax1 = plt.subplot(2, 3, 1)
# plt.plot(np.random.random(10))
# ax2 = plt.subplot(2, 3, 2)
# plt.plot(np.random.random(10))
# ax3 = plt.subplot(2, 3, 3)
# plt.plot(np.random.random(10))
# ax4 = plt.subplot(2, 1, 2)
# plt.plot(np.random.random(10))

f, ax = plt.subplots(2, 2)

ax[0, 0].plot(np.arange(0.5, 0.2))
ax[0, 0].grid()
plt.show()
