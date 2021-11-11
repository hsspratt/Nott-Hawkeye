# %%
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import math

# input camera angles
phi_xz = mp.radians(-30)
theta_x = mp.radians(45)

x = np.linspace(-100,100, 100)
Az_xz = mp.cot(theta_x)*x

n = 50

# plt.plot(x,Az_xz)
# plt.scatter(x[n],Az_xz[n], color='red', marker='x')
# plt.xlim(-100, 100);
# plt.ylim(0, 200);


Bz_xz = mp.tan(phi_xz)*(x+100)+100

# plt.figure()
# plt.plot(x,Bz_xz)
# plt.scatter(x[n], Bz_xz[n], color='red', marker='x')
# plt.xlim(-100, 100);
# plt.ylim(0, 200);

plt.figure()
plt.plot(x, Az_xz)
plt.plot(x, Bz_xz)
plt.xlim(-100, 100);
plt.ylim(0, 200);
# plt.scatter(-Bz_xz[n], x[n], color='red', marker='x')
# %%

