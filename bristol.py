""" numerical work on nuclear spins in a giant phase shift setup as in Bristol """

import numpy as np
import matplotlib.pyplot as plt

def r(w_QD,w_C,w,kappa,gamma,g):
    """ emitter detuning E_det,
    mode-in/out coupling kappa,
    lossy mode coupling kappa_s,
    emitter - cavity coupling g
    emitter lifetime gamma,
    cavity detuning C_det """
    r=1-kappa*(1j*(w_QD - w)+0.5*gamma)/ \
    ((1j*(w_QD - w)+0.5*gamma)*(1j*(w_C - w)+0.5*kappa)+g**2)
    return r

phase = lambda r: np.arctan2(r.imag,r.real)

class parameters:
    def __init__(self,params):
        (self.w_QD, self.w_C, self.kappa, self.gamma, self.g) = params

    def phase_method(self,w):
        refl=r(self.w_QD, self.w_C, w ,self.kappa, self.gamma, self.g)
        return phase(refl.imag,refl.real)

    def reflectivity(self,w):
        refl=r(self.w_QD, self.w_C, w ,self.kappa, self.gamma, self.g)
        return refl
    

def U(J, A, Om, w, T):
    """ Unitary evolution propagator for box model in |j m> |z> basis. After Ed's notes eq. 8 """
    
    dims=int(2*(2*J+1))
    
    E = lambda m,s: (m+0.5*s)*w - 0.25*A
    X = lambda J,m,s: A*np.sqrt(J*(J+1) - m*(m+s))
    Z = lambda m,s: s*(Om - w + A*(m + 0.5*s))
    N = lambda J,m,s: np.sqrt(X(J,m,s)**2 + Z(m,s)**2)

    a = lambda J,m,t: np.exp(-1j*E(m,1)*t)*(np.cos(0.5*N(J,m,1)*t) -
                                        1j*Z(m,1)/N(J,m,1)*np.sin(0.5*N(J,m,1)*t) )
    b = lambda J,m,t: -1j*np.exp(-1j*E(m,1)*t)*X(J,m,1)*np.sin(0.5*N(J,m,1)*t)/N(J,m,1)

    c = lambda J,m,t: np.exp(-1j*E(m,-1)*t)*(np.cos(0.5*N(J,m,-1)*t) -
                                        1j*Z(m,-1)/N(J,m,-1)*np.sin(0.5*N(J,m,-1)*t) )
    d = lambda J,m,t: -1j*np.exp(-1j*E(m,-1)*t)*X(J,m,-1)*np.sin(0.5*N(J,m,-1)*t)/N(J,m,-1)

    U_diag=[0]*dims

    for k in xrange(dims):
        if k % 2 ==0:
            m=J-k/2
            U_diag[k]= a(J,m,T)
        else:
            U_diag[k]= c(J,m,T)

    U_diag_up=[0]*(dims-1)
    U_diag_down=[0]*(dims-1)

    for k in xrange(dims-1):
        if k % 2 == 1:
            U_diag_up[k]= b(J,J - k/2 - 1,T)
            U_diag_down[k] = d(J,J - k/2,T)
    
    U = np.diag(U_diag) + np.diag(U_diag_up,k=1) + np.diag(U_diag_down,k=-1)
    return U

if __name__=='__main__':
    p_up=parameters([-2680., 0., 4100., .28, 38]) #w_QD, w_C, kappa, gamma, g
    p_down=parameters([-2720., 0., 4100., .28, 38.])
    x=np.linspace(-2730,-2670,100000)
    #g=lambda w: phase(p_up.reflectivity(w) / p_down.reflectivity(w))
    g=lambda w: abs(p_up.reflectivity(w))
    y=map(g,x)  
    plt.plot(x,y)
    plt.ylabel(r'$\theta_\uparrow - \theta_\downarrow$')
    plt.xlabel(r'$\omega - \omega_C$ [$\mu$eV]')
    plt.show()
