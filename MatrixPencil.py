'''
Matrix Pencil Method

@author: zbb
@date: 20190811
@updates:2020-09-17 
@ref: Tapan K. Sakar and Odilon Pereira, Using the Matrix Pencil Method to Estimate the Parameters of Sum of Complex Exponetials, 
IEEE Antennas and Propagation Magazine, Vol. 37, No. 1, February 1995.
'''

import numpy as np 
import matplotlib.pyplot as plt 

def _constructY(y, N, L):
    '''
    y: complex signal sequence.
    N: len(y)
    L: L<N, pencil parameter, N/3 < L < N/2 recommended. 
    return: constructed Y matrix, 
    [
        [y[0], y[1], ..., y[L-1]],
        [y[1], y[1, 0], ..., y[L]], 
        ...
        [y[N-L-1], y[N-L], ..., y[N-1]]
    ]
    (N-L)*(L+1) matrix. 
    '''
    Y = np.zeros((N-L, L+1), dtype=np.complex_)
    for k in range(N-L):
        Y[k, :] = y[k:(k+L+1)]
    return Y 

def _constructZM(Z, N):
    '''
    Z: 1-D complex array.
    return N*M complex matrix (M=len(Z)):
    [
        [1,  1,  1, ..., 1 ],
        [z[0], z[1], .., z[M-1]],
        ...
        [z[0]**(N-1), z[1]**(N-1), ..., z[M-1]**(N-1)]
    ]
    '''
    M = len(Z)
    ZM = np.zeros( (N, M), dtype=np.complex_) 
    for k in range(N):
        ZM[k, :] = Z**k 
    return ZM 

def _SVDFilter(Sp, p=3.0):
    '''
    Sp: 1-D normed eigenvalues of Y after SVD, 1-st the biggest
    p: precise ditigits, default 3.0. 
    return: M, M is the first integer that S[M]/S_max <= 10**(-p)
    '''
    Sm = np.max(Sp) 
    pp = 10.0**(-p)
    for m in range(len(Sp)):
        if Sp[m]/Sm <= pp:
            return m+1 
    return m+1 

def pencil(y, dt, M=None, p=8.0, Lfactor=0.40):
    '''
    Purpose:
      Complex exponential fit of a sampled complex waveform by Matrix Pencil Method.
    Authors: 
      Zbb
    Arguments:
      N    - number of data samples. ==len(y)       [INPUT]
      y    - 1-D complex array of sampled signal.   [INPUT]
      dt   - sample interval.                       [INPUT]
      M    - pencil parameter. 
             if None: use p to determin M.
             if given in range(0, Lfractor*N), then use it
             if given out of range, then use p to determin M.
      p    - precise digits of signal, default 8.0, corresponding to 10**(-8.0).
    Returns: (Z, R, M, (residuals, rank, s))
      Z    - 1-D Z array. 
      R    - 1-D R array.
      M    - M in use. 
      (residuals, rank, s)   - np.linalg.lstsq further results. 
    Method:
      y[k] = y(k*dt) = sum{i=0--> M} R[i]*( Z[i]**k ) 
      Z[i] = exp(si*dt)
    
    Comment: 
      To some extent, it is a kind of PCA method. 
    '''
    N = len(y)
    # better between N/3~N/2, pencil parameter:
    L = int(N*Lfactor)  
    # construct Y matrix (Hankel data matrix) from signal y[i], shape=(N-L, L+1):
    Y = _constructY(y, N, L)
    # SVD of Y: 
    _, S, V = np.linalg.svd(Y, full_matrices=True)
    #results: U.shape=(N-L, N-L), S.shape=(L+1, ), V.shape=(L+1, L+1)

    # find M: 
    if M is None:
        M = _SVDFilter(np.abs(S), p=p)
    elif M not in range(0, L+1):
        M = _SVDFilter(np.abs(S), p=p) 
    else: 
        pass
    # matrix primes based on M:
    #Vprime = V[0:M, :] # remove [M:] data set. only 0, 1, 2, ..., M-1 remains
    #Sprime = S[0:M]
    V1prime = V[0:M, 0:-1] # remove last column
    V2prime = V[0:M, 1:] # remove first column
    #smat = np.zeros((U.shape[0], M), dtype=np.complex_)
    #smat[:M, :M] = np.diag(Sprime)
    #Y1 = np.dot(U[:-1, :], np.dot(smat, V1prime))

    V1prime_H_MPinv = np.linalg.pinv(V1prime.T) # find V1'^+ , Moore-Penrose pseudoinverse 
    V1V2 = np.dot(V1prime_H_MPinv, V2prime.T) # construct V1V2 = np.dot(V1'^+, V2') 
    Z = np.linalg.eigvals(V1V2) # find eigenvalues of V1V2. Zs.shape=(M,)
    #print(V1V2.shape, Z)

    # find R by solving least-square problem: Y = np.dot(ZM, R)
    ZM = np.row_stack([Z**k for k in range(N)]) # N*M 
    R, residuals, rank, s = np.linalg.lstsq(ZM, y)
    return (Z, R, M, (residuals, rank, s))




if __name__ == '__main__':
    import random 
    import matplotlib.pyplot as plt 
    fig, axes = plt.subplots(1, 2) 
    '''
    @test 1: matrixpencil for complex, sampled data. 
    '''
    dt = 0.005
    x = np.arange(0.0, 1.0, dt,  dtype=np.float_) 
    N = len(x) # sapmle size  
    
    
    # sampled data: [complex]
    y = 10.0 * np.exp(-x*(5.04 + 8.0j)) + 1.0*np.exp(-x*(1.0 - 6.28J*10.3)) 
    y *= 1.0 + 0.05*np.array([random.random() for i in range(len(x))])
    
    # lets fit it:  
    Z, R, M, _ =  pencil(y, dt, M=6, p=3.0, Lfactor=0.33)
    print('@test2 result: ')
    print('expontial factors:\t', np.log(Z)/dt)
    print('amplitudes:\t', R)
    print('M =\t', M) 
    
    # display:
    y_from_result = np.zeros(shape=x.shape, dtype=np.complex_) 
    ks = np.linspace(0.0, N*1.0, N) 
    for idx in range(len(Z)):
        y_from_result += R[idx] * Z[idx]**ks 
        
    axes[0].scatter(x, np.real(y), c='k', s=2, label='real part of y')  
    axes[0].plot(x, np.real(y_from_result), 'r', linewidth=1, label='real part of y_result') 
    axes[0].set_xlabel('x') 
    tmp = np.abs(y) 
    axes[0].set_ylim((-np.amax(tmp)*1.0, np.amax(tmp)*1.5)) 
    axes[0].set_ylabel('y.re') 
    axes[0].set_title('@test1: real part compare') 
    axes[0].legend()
    
    axes[1].scatter(x, np.imag(y), c='k', s=2, label='real part of y')  
    axes[1].plot(x, np.imag(y_from_result), 'r', linewidth=1, label='imag part of y_result') 
    axes[1].set_xlabel('x') 
    axes[1].set_ylim((-np.amax(tmp)*1.0, np.amax(tmp)*1.5)) 
    axes[1].set_ylabel('y.imag') 
    axes[1].set_title('@test1: imag part compare') 
    axes[1].legend()
    
    '''
    @test2: matrixpencil for real, sampled data.  Find its envelope first. 
    '''

    plt.show() 


