"""
Wave Equation PDE Solver
created by Minha Jo, Sandro Henning, Murphy McDonough, and Emily Suh 
MA305 Project
12/1/2021
"""

##### import libraries #####
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.cm as cm


##### setup for plot #####
plt.rcParams['figure.figsize'] = [12,12]
plt.rcParams.update({'font.size': 18})

##### user input for domain & intitialization #####
c = float(input('Please enter Wave Speed: '))
#length for domain
L = int(input('Please input length of domain! (how much of the wave you would like to see, 1-100): ')) 
#number of discretization points
N = int(input('How close do you want to see the graph show the continuous wave? 1000 = very close, 10 = not that close): '))													
#step size
dx = L/N
#space
x = np.arange(-L/2, L/2, dx)

##### setting amount of discrete waves #####
k = 2*np.pi*np.fft.fftfreq(N, d = dx)

##### initial condition of wave #####
u0 = 1/np.cosh(x) 
u0hat = np.fft.fft(u0) 

##### split complex numbers #####
u0hat_split = np.concatenate((u0hat.real,u0.imag))

##### put into fourier #####
dt = 0.025
t = np.arange(0,100*dt,dt) 

##### set up wave equation #####
def waveEq(uhat_split,t,k,c):
    uhat = uhat_split[:N] + (1j) * uhat_split[N:]
    d_uhat = -c*(1j)*k*uhat
    d_uhat_split = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')
    return d_uhat_split
  
##### put uhat into ode function #####
uhat_split = odeint(waveEq, u0hat_split, t, args=(k,c))
uhat = uhat_split[:, :N] + (1j) * uhat_split[:, N:]


##### spatial domain equation backup #####
"""
def spatialEq(u,t,k,c):
    uhat = np.fft.fft(u)
    d_uhat = (1j)* k * uhat
    du = np.fft.ifft(d_uhat).real
    dudt = -c*du
    return dudt
    
    
u = odeint(spatialEq, u0, t, args=(k,c))
"""
##### inverse fast fourier transform for spatial domain #####

u = np.zeros_like(uhat) 

for i in range(len(t)):
		u[i,:] = np.fft.ifft(uhat[i,:])
    
u = u.real

##### show 3D waves simulation #####
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')


u_plot = u[0:-1:10,:]
for i in range(u_plot.shape[0]):
    y = i*np.ones(u_plot.shape[1])
    ax.plot(x,y,u_plot[i,:], color = cm.jet(i*20))
    
##### show top down view #####
plt.figure()
plt.imshow(np.flipud(u), aspect = 8)
plt.axis('off')
plt.set_cmap('jet_r')
plt.show()
