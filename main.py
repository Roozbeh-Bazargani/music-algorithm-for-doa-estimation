'''
Written by Roozbeh Bazargani
Date 6/18/2019
'''

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('test.mat')
x = mat['x']
covar = np.cov(x)
U, sigma, V = np.linalg.svd(covar)
# eigval, eigvec = np.linalg.eig(covar)
# sigma = abs(eigval)
s0 = sigma[0]
k = 0
for i in range(1,len(sigma)):
    if s0 / sigma[i] > 10:
        k = i
        break


# U1 = U[:,k+1:]
# V1 = V[k+1:,:]
# sigma1 = sigma[k+1:,k+1:]
# Un = U1 * sigma1 * V1
# print(U[k+1:,k+1:].shape,V[k+1:,k+1:].shape)
# print(np.column_stack([np.diag(sigma[k+1:]),np.zeros([sigma.shape[0] - (k + 1),V.shape[0] - U.shape[0]])]).shape)
# print([np.diag(sigma),np.zeros([sigma.shape[0],V.shape[0] - U.shape[0]])])
# U0 = np.matrix(U) * np.column_stack([np.diag(sigma),np.zeros([sigma.shape[0],V.shape[0] - U.shape[0]])]) * np.matrix(V)
# Un = np.matrix(U[k+1:,k+1:]) \
#      * np.column_stack([np.diag(sigma[k+1:]),np.zeros([sigma.shape[0] - (k + 1),V.shape[0] - U.shape[0]])])\
#      * np.matrix(V[k+1:,k+1:])
Un = np.matrix(U[:,k:])
# print(Un.shape)

# print(k)

theta = np.arange(-90,90.1,0.25)
# theta = sympy.symbols('theta')
# print(sympy.cos(theta).subs(theta,theta1))
# print(theta.shape[0])
Ptheta = np.zeros(theta.shape[0])

for i in range(theta.shape[0]):
    sin = [np.exp(-1j*i1*np.pi*np.sin(theta[i]*np.pi/180.)) for i1 in range(Un.shape[0])]
    ath = np.matrix(np.row_stack(sin))
    Ptheta[i] = 1 / abs(ath.getH()*Un*Un.getH()*ath)
# print(Ptheta)



plt.plot(theta, Ptheta)

plt.xlabel(r'theta', fontsize=16)
plt.ylabel(r'P(theta)',fontsize=16)
plt.title(r"Music",
fontsize=16, color='gray')
# Make room for the ridiculously large title.
plt.subplots_adjust(top=0.8)

plt.show()
