""" plot Zeno time vs QD radius r_c and wavefunction
    decay constant R. """

import numpy as np
import matplotlib.pyplot as plt

m = lambda i,j,r_c,R: np.exp(-R*(i**2 + j**2))*int(i**2+j**2<r_c**2)

def tz(r_c,R):
    """ Zeno time for QD radius r_c and wavefunction
    decay constant R. """
    x=range(-r_c,r_c+1)
    y=range(-r_c,r_c+1)
    xx,yy=np.meshgrid(x,y)
    e_vec1=np.vectorize(lambda i,j: m(i,j,r_c,R))
    A=5.e10/6.582*e_vec1(xx,yy)/e_vec1(xx,yy).sum()
    tz_sq_inv=(10./4.)*np.square(A)
    tz=1/np.sqrt(tz_sq_inv.sum())
    return tz

xR=np.linspace(0.,1.e-1,100)
x_rc=range(15,36)
yR=map(lambda x: tz(25,x),xR)
y_rc=map(lambda x:tz(x,1.e-2),x_rc)
plt.plot(xR,yR)
plt.xlabel('wave-function decay constant R. QD radius is 25 lattice constants')
plt.ylabel(r'$\tau_Z$ [s]')
"""
plt.plot(x_rc,y_rc)
plt.xlabel('radius of quantum dot in lattice constants. R=0.1')
plt.ylabel(r'$\tau_Z$ [s]')
"""
plt.show()


