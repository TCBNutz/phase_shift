""" numerical work on nuclear spins in a giant phase shift setup as in Bristol """

import numpy as np
import matplotlib.pyplot as plt
import random
from U_box import U

ir2 = 1 / np.sqrt(2)
H = np.array([1., 0.])
V = np.array([0., 1.])
H_proj=np.array([[1,0],[0,0]])
V_proj=np.array([[0,0],[0,1]])
hbar=6.582119514e-10 #ueV s

def TMany(x):
    """ tensor product of many matrices """
    return reduce(np.kron, x)

def DMany(x):
    """ dot product of matrices in list x """
    return reduce(np.dot, x)

phase = lambda r: np.arctan2(r.imag,r.real)

p_shift = lambda theta: 0.5*(np.array([[1,-1j],[1j,1]]) + np.exp(1j*theta)*np.array([[1,1j],[-1j,1]]))

adj=lambda x: np.conj(np.transpose(x))

max_mix=lambda J: np.eye(2*(2*J+1))/(2*(2*J+1))

def r(w_QD,w_C,w,kappa,gamma,g):
    """ emitter energy w_QD,
    cavity resonance w_C,
    laser frequency w,
    cavity linewidth kappa,
    emitter lifetime gamma,
    emitter - cavity coupling g """
    r=1-kappa*(1j*(w_QD - w)+0.5*gamma)/ \
    ((1j*(w_QD - w)+0.5*gamma)*(1j*(w_C - w)+0.5*kappa)+g**2)
    return r

def CP(p):
    """ makes phase shift conditional on nuclear z-projection.
    j is the size of the box model block, p is an instance of the
    parameters class. """
    U=np.zeros((2*(2*p.J+1),2*(2*p.J+1)),dtype = np.complex128)
    for k in xrange(p.J,-p.J-1,-1):
        U += np.kron(np.diag([0]*(p.J-k) + [1] + [0]*(p.J+k)),p_shift(p.phase_diff(k)))
    U_up=TMany([H_proj,np.eye(2*(2*p.J+1))])
    U_down=np.kron(V_proj,U)
    return U_up+U_down

def U_Box(p,t):
    """ Box model Hamiltonian for parameters instance p and time t [us]. """
    return U(p.J,p.A,p.Zeeman,p.Zeeman/1000,t/hbar)

def R(r_en,p):
    """ performs probabilistic measurement on the photon that has interacted
    with electron-nuclear state r_en. p is an instance of parameters."""
    J=(r_en.shape[0]-1)/4
    id_en=np.eye(2*(2*J+1))
    H_big , V_big = np.kron(id_en,H_proj) , np.kron(id_en,V_proj)
    r_enp=np.kron(r_en,V_proj)
    r_r_enp=DMany([CP(p),r_enp,adj(CP(p))])
    p_H=np.trace(np.dot(H_big,r_r_enp))
    if random.random() < p_H:
        outcome='h'
        r_out=DMany([np.kron(id_en,H),r_r_enp,np.kron(id_en,np.array([[1],[0]]))])/p_H
    else:
        outcome='v'
        r_out=DMany([np.kron(id_en,V),r_r_enp,np.kron(id_en,np.array([[0],[1]]))])/(1-p_H)
    return [r_out,outcome]

class parameters:
    def __init__(self):
        self.J=3
        self.w_C=2700.
        self.w=0.
        self.kappa=4100.
        self.gamma=0.28
        self.g=38.
        self.A=5.
        self.Zeeman=50.

    def phase_diff(self,m):
        r_down=r(-0.5*m*self.A,self.w_C, self.w ,self.kappa, self.gamma, self.g)
        r_up=r(self.Zeeman+0.5*m*self.A,self.w_C, self.w ,self.kappa, self.gamma, self.g)
        return phase(r_down/r_up)

if __name__=='__main__':
    p=parameters()
    time_obs=1000 #us, photon detection on average every 1 us
    times=[time_obs*random.random() for i in xrange(time_obs)] # 100 random times
    times.sort()
    intervals=[j-i for i, j in zip(times[:-1], times[1:])] 
    results=[]
    state=max_mix(p.J)
    for i in intervals:
        state,outcome=R(DMany([U_Box(p,i),state,adj(U_Box(p,i))]),p)
        results.append(outcome)

    """
    # plot phase shift over QD energy
    x=np.linspace(-3,+3,10000)
    g=lambda w: p.phase_diff(w)
    y=map(g,x)  
    plt.plot(x,y)
    plt.ylabel(r'$\theta_\uparrow - \theta_\downarrow$')
    plt.xlabel('m')
    plt.show()
    """
