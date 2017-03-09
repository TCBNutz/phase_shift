import numpy as np
import matplotlib.pyplot as plt


f=lambda t: np.exp(-0.5*g*t)*np.cos(0.5*np.sqrt(16.*Om**2 - g**2 + 0.j)*t)
x=np.linspace(0,5,500)

"""
g,Om = 0.,1.
plt.plot(x,map(f,x))


g,Om = 1.,1.
plt.plot(x,map(f,x))


g,Om = 2.,1.
plt.plot(x,map(f,x))
"""

g,Om = 200.,1.
plt.plot(x,np.real(map(f,x)))

plt.show()
