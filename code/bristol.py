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

IZ_var = lambda p,state: np.real(np.trace(DMany([IZ(p.J),IZ(p.J),state])) \
                                 - np.trace(np.dot(IZ(p.J),state))**2) # variance
IZ_stdev=lambda p,state: np.sqrt(IZ_var(p,state)) # standard deviation

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
        outcome=1 #h
        r_out=DMany([np.kron(id_en,H),r_r_enp,np.kron(id_en,np.array([[1],[0]]))])/p_H
    else:
        outcome=0 #v
        r_out=DMany([np.kron(id_en,V),r_r_enp,np.kron(id_en,np.array([[0],[1]]))])/(1-p_H)
    return [r_out,outcome]

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
        pre_stdev=IZ_stdev(p,state)
        state,outcome=R(DMany([U_Box(p,i),state,adj(U_Box(p,i))]),p)
        post=IZ_eval(p,state) # nuclear polarization after
        post_stdev=IZ_stdev(p,state)
        results.append([outcome,pre,pre_stdev,post,post_stdev])
    return [results,times]

def timebin_scatter(Run,tbin):
    """ make scatter plot with horizontal (cross-polarized) counts on x and
    vertical counts on y. """
    tbin=30 # length of timebin in number of photons
    t=len(Run[1])
    outcomes=map(lambda x:x[0],Run[0])
    averages=np.zeros(len(outcomes))
    lobe_plot_coords=np.zeros((t/tbin,2))
    for i in xrange(t/tbin-1):
        l=outcomes[i*tbin:(i+1)*tbin]
        average=sum(l)/len(l)
        averages[i*tbin:(i+1)*tbin]=[average]*tbin
        lobe_plot_coords[i][0],lobe_plot_coords[i][1]= \
            np.bincount(l)[0],tbin-np.bincount(l)[0]
    a=np.transpose(lobe_plot_coords[:-1])
    return averages,a

def run_plot(Run,tbin=30):
    averages,a=timebin_scatter(Run,tbin)
    y1=map(lambda w: w[1],Run[0]) #nuke pol
    y2=map(lambda w: w[2],Run[0]) #stdev
    plt.plot(Run[1][1:],y1,'r') #pol
    plt.plot(Run[1][1:],y2,'b') #stdev
    plt.plot(Run[1][1:],averages,'c') # outcomes
    plt.ylabel(r'red: $<I_z>$, blue: $\sigma (I_z)$, cyan: photon polarization')
    plt.xlabel('time')
    #plt.scatter(a[1],a[0],s=70, alpha=0.07); plt.show()
    plt.show()

def plot_phaseshift(p,J):
    """ plots phaseshift for down electron. p is instance of parameters. """
    m=np.linspace(-J,J,2*J+1).tolist()
    phases=map(lambda x: (phase(p.phases(x)[2]/p.phases(x)[3]))/np.pi,m)
    return m,phases

class parameters:
    def __init__(self):
        self.J=10
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

if __name__=='__main__':
    #t=200 #time of the run in us
    #Run=run(t)
    #run_plot(Run)
    p=parameters()
    J=50
    m,phases=plot_phaseshift(p,J);
    m.reverse()
    phases.reverse()
    proj=np.zeros(len(m))
    for i in xrange(len(phases)):
        if abs(phases[i]) > 0.2:
            proj[i]=1.
    proj=np.kron(np.diag(proj),np.eye(2))
    g=lambda x: np.real(np.trace(np.dot((np.eye(2*(2*J+1))-proj),x)))
    init=np.eye(202)/202.#np.diag([0]*94 + [1] + [0]*107)
    gt=lambda t: g(DMany([U(J,p.A,p.Zeeman,0.,t),init,adj(U(J,p.A,p.Zeeman,0.,t))]))
    plt.plot(np.linspace(1e-4,100),map(gt,np.linspace(10,1000)))
    plt.show()
