""" numerical work on nuclear spins in a giant phase shift setup as in Bristol """

import numpy as np
import matplotlib.pyplot as plt
import random
from U_box import U, Iz

ir2 = 1 / np.sqrt(2)
H = np.array([1., 0.])
V = np.array([0., 1.])
H_proj=np.array([[1,0],[0,0]])
V_proj=np.array([[0,0],[0,1]])
R_proj=0.5*np.array([[1,-1j],[1j,1]])
L_proj=0.5*np.array([[1,1j],[-1j,1]])
hbar=6.582119514e-10 #ueV s

def TMany(x):
    """ tensor product of many matrices """
    return reduce(np.kron, x)

def DMany(x):
    """ dot product of matrices in list x """
    return reduce(np.dot, x)

phase = lambda r: np.arctan2(r.imag,r.real)

p_shift = lambda rl: rl[0]*R_proj + rl[1]*L_proj

adj=lambda x: np.conj(np.transpose(x))

max_mix=lambda J: np.eye(2*(2*J+1))/(2*(2*J+1))

IZ=lambda J: np.kron(Iz(J),np.eye(2))

IZ_eval = lambda p,state: np.real(np.trace(np.dot(IZ(p.J),state))) # expectation value of I_z. p is parameters instance

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
    U_up=np.zeros((4*(2*p.J+1),4*(2*p.J+1)),dtype = np.complex128)
    U_down=np.zeros((4*(2*p.J+1),4*(2*p.J+1)),dtype = np.complex128)
    for k in xrange(p.J,-p.J-1,-1):
        U_up += TMany([np.diag([0]*(p.J-k) + [1] + \
                               [0]*(p.J+k)),H_proj,p_shift(p.phases(k)[:2])])
        U_down += TMany([np.diag([0]*(p.J-k) + [1] + \
                                 [0]*(p.J+k)),V_proj,p_shift(p.phases(k)[2:])])
    return U_up+U_down

def U_Box(p,t):
    """ Box model Hamiltonian for parameters instance p and time t [us]. """
    return U(p.J,p.A,p.Zeeman,p.Zeeman/1000,t/hbar)

def trace_over_elnuc(rho_enp):
    """ traces over all but the innermost qubit """
    photon=np.zeros((2,2),dtype=np.complex128)
    d=len(rho_enp)/2
    for i in xrange(d):
        p=np.matrix(np.zeros((1,d)))
        p[0,i]=1
        p=np.kron(p,np.asmatrix(np.identity(2)))
        photon += DMany([p,rho_enp,p.H])
    return photon

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
        self.A=.34 # pi phase shift at m=-2
        self.Zeeman=50.

    def phases(self,m):
        """ exp(phase(r)) given nuclear spin projection m """
        r_DL=r(-0.5*m*self.A,self.w_C, self.w ,self.kappa, self.gamma, self.g)
        r_DR=r(-0.5*m*self.A,self.w_C, self.w ,self.kappa, self.gamma, 0.)
        r_UL=r(0.5*m*self.A,self.w_C, self.w ,self.kappa, self.gamma, 0.)        
        r_UR=r(self.Zeeman+0.5*m*self.A,self.w_C, self.w ,self.kappa, self.gamma, self.g)
        return map(lambda x:x/abs(x),[r_UR,r_UL,r_DR,r_DL])

def run(t):
    """ input length of run in us """
    p=parameters()
    times=[t*random.random() for i in xrange(t)] # random times
    times.sort()
    intervals=[j-i for i, j in zip(times[:-1], times[1:])] 
    results=[]
    state=max_mix(p.J)
    for i in intervals:
        pre=IZ_eval(p,state) # nuclear polarization before
        state,outcome=R(DMany([U_Box(p,i),state,adj(U_Box(p,i))]),p)
        post=IZ_eval(p,state) # nuclear polarization after
        results.append([outcome,pre,post])
    return [results,times]


if __name__=='__main__':
    
    Run=run(2000)
    
    g=lambda w: w[1]
    y=map(g,Run[0])  
    plt.plot(Run[1][1:],y)
    plt.ylabel(r'$I_z$ pre')
    plt.xlabel('time')
    plt.show()
    
    """
    # plotting phase shift
    p=parameters()
    x=np.linspace(-10,10,21)
    #y1=map(lambda x: phase(p.phases(x)[0]/p.phases(x)[1]),x)
    y2=map(lambda x: (phase(p.phases(x)[2]/p.phases(x)[3]))/np.pi,x)
    #plt.plot(x,y1)
    plt.plot(x,y2,'bo')
    plt.show()
    """
