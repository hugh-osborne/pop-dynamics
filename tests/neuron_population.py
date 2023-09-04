import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import poisson

import cupy as cp

import time

from popdynamics.popsolver import Solver
from popdynamics.fastpopsolver import FastSolver
from popdynamics.visualiser import Visualiser

use_monte_carlo = True
use_cpu_solver = True
use_gpu_solver = False
plot_output = True

def cond(y):
    E_l = -70.6
    E_e = 0.0
    E_i = -75
    C = 281
    g_l = 3.0
    tau_e =2.728
    tau_i = 10.49

    threshold = -50.4
    reset = -70.6

    v = y[0]
    w = y[1]
    u = y[2]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (v - E_i)) / C
    w_prime = -(w) / tau_e
    u_prime = -(u) / tau_i

    nv = v + v_prime

    if nv > threshold:
        nv = reset

    return [nv, w + 0.1*w_prime, u + 0.1*u_prime]

v_res = 100
w_res = 100
u_res = 100
I_res = 101

v_max = -40.0
v_min = -80.0
w_max = 50.0
w_min = -5.0
u_max = 50.0
u_min = -5.0

# Set up the starting distribution
v = np.linspace(v_min, v_max, v_res)
w = np.linspace(w_min, w_max, w_res)
u = np.linspace(u_min, u_max, u_res)

#[-70.6, 0.001, 0.001]
# Unfortunately stats.norm doesn't provide a nice pmf approximation of the pdf. 
# So let's just do that ourselves without breaking our backs by multiplying across by the discretisation width and normalising
vpdf = [a * ((v_max-v_min)/v_res) for a in norm.pdf(v, -70.6, 0.1)]
wpdf = [a * ((w_max-w_min)/w_res) for a in norm.pdf(w, 0.0, 0.1)]
updf = [a * ((u_max-u_min)/u_res) for a in norm.pdf(u, 0.0, 0.1)]

vpdf = [a / sum(vpdf) for a in vpdf]
wpdf = [a / sum(wpdf) for a in wpdf]
updf = [a / sum(updf) for a in updf]

# Poisson inputs

w_rate = 4
epsp = 0.5
wI_max_events = 5
wI_min_events = -5
wI_max = wI_max_events*epsp
wI_min = wI_min_events*epsp
epsps = np.linspace(wI_min, wI_max, I_res)
wI_events = np.linspace(wI_min_events, wI_max_events, I_res)
wIpdf_final = [0 for a in wI_events]
for i in range(len(wI_events)-1):
    if (int(wI_events[i]) < int(wI_events[i+1])) or (wI_events[i] < 0 and wI_events[i+1] >= 0): # we have just passed a new event
        e = int(wI_events[i+1])
        if e <= 0:
            e = int(wI_events[i])
        diff = wI_events[i+1] - wI_events[i]
        lower_prop = (int(wI_events[i+1]) - wI_events[i]) / diff 
        upper_prop = 1.0 - lower_prop
        wIpdf_final[i] += poisson.pmf(e, w_rate*0.1) * lower_prop
        wIpdf_final[i+1] += poisson.pmf(e, w_rate*0.1) * upper_prop
wIpdf = wIpdf_final


wI_res = int((wI_max-wI_min) / ((w_max-w_min)/w_res))
if (wI_res % 2) == 0:
    wI_res += 1
pymiind_wI = [0 for a in range(wI_res)]
ratio = I_res / wI_res
val_counter = 0.0
target_counter = 0
for i in range(I_res):
    if i+1 > ratio*(target_counter+1):
        val_counter += wIpdf[i] * (((ratio*(target_counter+1)) - i))
        pymiind_wI[target_counter] = val_counter
        val_counter = wIpdf[i] * (1.0-(((ratio*(target_counter+1)) - i)))
        target_counter += 1
    else:
        val_counter += wIpdf[i]

u_rate = 2
ipsp = 0.5
uI_max_events = 5
uI_min_events = -5
uI_max = wI_max_events*ipsp
uI_min = wI_min_events*ipsp
ipsps = np.linspace(uI_min, uI_max, I_res)
uI_events = np.linspace(uI_min_events, uI_max_events, I_res)
uIpdf_final = [0 for a in uI_events]
for i in range(len(uI_events)-1):
    if (int(uI_events[i]) < int(uI_events[i+1])) or (uI_events[i] < 0 and uI_events[i+1] >= 0): # we have just passed a new event
        e = int(uI_events[i+1])
        if e <= 0:
            e = int(uI_events[i])
        diff = uI_events[i+1] - uI_events[i]
        lower_prop = (int(uI_events[i+1]) - uI_events[i]) / diff 
        upper_prop = 1.0 - lower_prop
        uIpdf_final[i] += poisson.pmf(e, u_rate*0.1) * lower_prop
        uIpdf_final[i+1] += poisson.pmf(e, u_rate*0.1) * upper_prop
uIpdf = uIpdf_final

uI_res = int((uI_max-uI_min) / ((u_max-u_min)/u_res))
if (uI_res % 2) == 0:
    uI_res += 1
pymiind_uI = [0 for a in range(uI_res)]
ratio = I_res / uI_res
val_counter = 0.0
target_counter = 0
for i in range(I_res):
    if i+1 > ratio*(target_counter+1):
        val_counter += uIpdf[i] * (((ratio*(target_counter+1)) - i))
        pymiind_uI[target_counter] = val_counter
        val_counter = uIpdf[i] * (1.0-(((ratio*(target_counter+1)) - i)))
        target_counter += 1
    else:
        val_counter += uIpdf[i]

# Initialise the monte carlo neurons
if use_monte_carlo:
    mc_neurons = np.array([[norm.rvs(-70.6, 0.1, 1)[0],norm.rvs(0.0, 0.1, 1)[0],norm.rvs(0.0, 0.1, 1)[0]] for a in range(5000)])


# CPU solver
dims = 3
cell_widths = [(v_max-v_min)/v_res, (w_max-w_min)/w_res, (u_max-u_min)/u_res]

initial_dist = np.ndarray((v_res, w_res, u_res))
for cv in range(v_res):
    for cw in range(w_res):
        for cu in range(u_res):
            initial_dist[cv,cw,cu] = vpdf[cv]*wpdf[cw]*updf[cu]
            
if use_cpu_solver:
    perf_time = time.perf_counter()
    vis = Visualiser()
    vis.setupVisuliser()
    solver = Solver(cond, initial_dist, np.array([v_min,w_min,u_min]), cell_widths, 0.00000001, vis)
    solver.addNoiseKernel(pymiind_wI, 1)
    solver.addNoiseKernel(pymiind_uI, 2)
    print("CPU Setup time:", time.perf_counter() - perf_time)


# GPU solver

if use_gpu_solver:
    perf_time = time.perf_counter()
    gpu_solver = FastSolver(cond, initial_dist, [v_min, w_min, u_min], [v_max-v_min, w_max-w_min, u_max-u_min], [v_res, w_res, u_res])
    gpu_solver.addNoiseKernel(pymiind_wI, 1)
    gpu_solver.addNoiseKernel(pymiind_uI, 2)
    print("GPU Setup time:", time.perf_counter() - perf_time)

vis.beginRendering()

perf_time = time.perf_counter()
for iteration in range(101):

    # CPU Solver
    if use_cpu_solver:
        solver.updateDeterministic()
        solver.applyNoiseKernels()
        solver.draw()

    # GPU Solver
    if use_gpu_solver:
        gpu_solver.updateDeterministic()
        gpu_solver.applyNoiseKernels()

    # Also run the monte carlo simulation 

    if use_monte_carlo:
        fired_count = 0

        for nn in range(len(mc_neurons)):
            mc_neurons[nn] = cond(mc_neurons[nn])
        
            if (mc_neurons[nn][0] > -50.0):
                mc_neurons[nn][0] = -70.6
                fired_count+=1
                
            mc_neurons[nn][1] += epsp*poisson.rvs(w_rate*0.1) # override w
            mc_neurons[nn][2] += ipsp*poisson.rvs(u_rate*0.1) # override u  
            
    if plot_output and (iteration % 1 == 0) :
        # Plot Monte Carlo
        fig, ax = plt.subplots(2,2)

        if use_monte_carlo:
            ax[0,0].hist(mc_neurons[:,0], density=True, bins=v_res, range=[v_min,v_max], histtype='step')
            ax[0,1].hist(mc_neurons[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
            ax[1,0].hist(mc_neurons[:,2], density=True, bins=u_res, range=[u_min,u_max], histtype='step')
        
        # Plot CPU Solver marginals
        if use_cpu_solver:
            mpos, marginals = solver.calcMarginals()

            marginals[0] = [a / (cell_widths[0]) for a in marginals[0]]
            marginals[1] = [a / (cell_widths[1]) for a in marginals[1]]
            marginals[2] = [a / (cell_widths[2]) for a in marginals[2]]

            ax[0,0].scatter(mpos[0], marginals[0])
            ax[0,1].scatter(mpos[1], marginals[1])
            ax[1,0].scatter(mpos[2], marginals[2])

        # Plot GPU Solver marginals
        if use_gpu_solver:
            mpos, marginals = gpu_solver.calcMarginals()

            marginals[0] = [a / (cell_widths[0]) for a in marginals[0]]
            marginals[1] = [a / (cell_widths[1]) for a in marginals[1]]
            marginals[2] = [a / (cell_widths[2]) for a in marginals[2]]

            ax[0,0].plot(mpos[0], marginals[0])
            ax[0,1].plot(mpos[1], marginals[1])
            ax[1,0].plot(mpos[2], marginals[2])

        fig.tight_layout()
        plt.show()

print("Total simulation time:", time.perf_counter() - perf_time)

vis.endRendering()