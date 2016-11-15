import numpy as np

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

def Iz(J):
    """ total nuclear spin Z-operator """
    if J==0:
        return np.array([1.])
    else:
        diagonal=np.arange(J,-J-1,-1)
        return np.diag(diagonal)
