from distutils.ccompiler import show_compilers
from pycausaljazz import pycausaljazz as cj
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import poisson
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class Solver:
    def __init__(self, initial_distribution, _base, _cell_widths):
        self.dims = _base.shape[0]
        self.cell_widths = _cell_widths

        self.cell_buffers = [{},{}]

        self.cell_base = np.zeros(self.dims)
        self.current_buffer = 0
        self.remove_sum = 1.0

        self.noise_kernels = []

        first_cell = True
        cell_base_coords = np.zeros(self.dims)
        for idx, val in np.ndenumerate(initial_distribution):
            if val > 0.0:
                if first_cell:
                    cell_base_coords = idx
                    self.cell_base = _base + (np.multiply(idx,_cell_widths))
                    first_cell = False
                self.cell_buffers[self.current_buffer][tuple((idx-cell_base_coords).tolist())] = [val, []]

    def addNoiseKernel(self, kernel):


    def calcTransitions(self, centroid, stepped_centroid, coord, d=0, target_coord=[], mass=1.0):
        if len(target_coord) == len(coord):
            return [(mass, target_coord)]

        diff = stepped_centroid[d] - centroid[d]
        cell_lo = coord[d] + int(diff / cell_widths[d])
        cell_hi = cell_lo + 1
        prop_lo = 0.0
        if diff < 0.0: # actually, diff is negative so cell_lo is the upper cell
            cell_hi = cell_lo - 1
            prop_lo = ((diff % cell_widths[d]) / cell_widths[d])
        else:
            prop_lo = 1.0 - ((diff % cell_widths[d]) / cell_widths[d])
        prop_hi = 1.0 - prop_lo
    
        return self.calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_lo], mass*prop_lo) + calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_hi], mass*prop_hi)

    def updateCell(self, relative, new_cell_dict, transition, coord, mass, func):
        t = [a for a in transition[1]]
        if relative:
            for d in range(len(coord)):
                t[d] = coord[d] + t[d]

        if tuple(t) not in new_cell_dict.keys():
            new_cell_dict[tuple(t)] = [0.0,[]]
            centroid = [0 for a in range(len(coord))]

            for d in range(len(coord)):
                centroid[d] = self.cell_base[d] + ((t[d]+0.5)*self.cell_widths[d])

            stepped_centroid = func(centroid)
            new_cell_dict[tuple(t)][1] = calcTransitions(centroid, stepped_centroid, t)
    
        new_cell_dict[tuple(t)][0] += mass*transition[0]

    def calcCellCentroid(self, coords):
        centroid = [0 for a in range(self.dims)]

        for d in range(self.dims):
            centroid[d] = self.cell_base[d] + ((coords[d]+0.5)*self.cell_widths[d])

        return centroid

    def calcMarginals(self):
        vs = [{} for d in range(self.dims)]
        for c in self.cell_buffers[self.current_buffer]:
            for d in range(self.dims):
                if c[d] not in vs[d]:
                    vs[d][c[d]] = self.cell_buffers[self.current_buffer][c][0]
                else:
                    vs[d][c[d]] += self.cell_buffers[self.current_buffer][c][0]

        final_vs = [[] for d in range(self.dims)]
        final_vals = [[] for d in range(self.dims)]

        for d in range(self.dims):
            for v in vs[d]:
                final_vs[d] = final_vs[d] + [self.cell_base[d] + (self.cell_widths[d]*(v))]
                final_vals[d] = final_vals[d] + [vs[d][v]]

        return final_vs, final_vals


# 3D cond as it should be in its entirety - but we're going to split this into separate functions
def cond(y):
    E_l = -70.6
    E_e = 0.0
    E_i = -75
    C = 281
    g_l = 0.03
    tau_e =2.728
    tau_i = 10.49

    v = y[0]
    w = y[1]
    u = y[2]

    v_prime = (-g_l*(v - E_l) - w * (v - E_e) - u * (v - E_i)) / C
    w_prime = -(w) / tau_e
    u_prime = -(u) / tau_i

    return [v + 0.1*v_prime, w + 0.1*w_prime, u + 0.1*u_prime]

res = 100
v_res = 100
w_res = 100
u_res = 100
I_res = 300

v_max = -40.0
v_min = -80.0
w_max = 50.0 #25.0
w_min = -5.0 #-1.0
u_max = 50.0
u_min = -5.0

# Set up the starting distribution
v = np.linspace(v_min, v_max, v_res)
w = np.linspace(w_min, w_max, w_res)
u = np.linspace(u_min, u_max, u_res)

points = []
for x in range(u_res):
    points_col = []
    for y in range(w_res):
        points_dep = []
        for z in range(v_res):
            points_dep = points_dep + [(x,y,z)]
        points_col = points_col + [points_dep]
    points = points + [points_col]
    
#[-70.6, 0.001, 0.001]
# Unfortunately stats.norm doesn't provide a nice pmf approximation of the pdf. 
# So let's just do that ourselves without breaking our backs by multiplying across by the discretisation width and normalising
vpdf = [a * ((v_max-v_min)/v_res) for a in norm.pdf(v, -70.6, 0.1)]
wpdf = [a * ((w_max-w_min)/w_res) for a in norm.pdf(w, 0.0, 0.1)]
updf = [a * ((u_max-u_min)/u_res) for a in norm.pdf(u, 0.0, 0.1)]

vpdf = [a / sum(vpdf) for a in vpdf]
wpdf = [a / sum(wpdf) for a in wpdf]
updf = [a / sum(updf) for a in updf]

v0 = cj.newDist([v_min],[(v_max-v_min)],[v_res],[a for a in vpdf])
w0 = cj.newDist([w_min],[(w_max-w_min)],[w_res],[a for a in wpdf])
u0 = cj.newDist([u_min],[(u_max-u_min)],[u_res],[a for a in updf])

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
wI = cj.newDist([wI_min], [(wI_max-wI_min)], [I_res], [a for a in wIpdf])

# For pymiind, I kernel cell size must match the joint cell size
wI_res = int((wI_max-wI_min) / ((w_max-w_min)/w_res))+1
pymiind_wI = [0 for a in range(wI_res)]
ratio = I_res / wI_res
val_counter = 0.0
pos_counter = 0
for i in range(I_res):
    pos_counter += 1
    if pos_counter > ratio:
        pos_counter = 0
        val_counter += wIpdf[i] * (i % ratio)
        pymiind_wI[int(i/ratio)] = val_counter
        val_counter = wIpdf[i] * (1.0-(i % ratio))
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
uI = cj.newDist([uI_min], [(uI_max-uI_min)], [I_res], [a for a in uIpdf])

uI_res = int((uI_max-uI_min) / ((u_max-u_min)/u_res))+1
pymiind_uI = [0 for a in range(uI_res)]
ratio = I_res / uI_res
val_counter = 0.0
pos_counter = 0
for i in range(I_res):
    pos_counter += 1
    if pos_counter > ratio:
        pos_counter = 0
        val_counter += uIpdf[i] * (i % ratio)
        pymiind_uI[int(i/ratio)] = val_counter
        val_counter = uIpdf[i] * (1.0-(i % ratio))
    else:
        val_counter += uIpdf[i]

# CPU MIIND

dims = 3
cell_widths = [(v_max-v_min)/v_res, (w_max-w_min)/w_res, (u_max-u_min)/u_res]

cell_dict = [{},{}]

cell_base = [0.0 for a in range(dims)]
first_cell = True
current_dict = 0
remove_sum = 1.0

cell_base_coords = [0,0,0]
for cv in range(v_res):
    for cw in range(w_res):
        for cu in range(u_res):
            val = vpdf[cv]*wpdf[cw]*updf[cu]
            if val > 0.0:
                if first_cell:
                    cell_base_coords = [cv,cw,cu]
                    cell_base = [v_min + cv*cell_widths[0], w_min + cw*cell_widths[1], u_min + cu*cell_widths[2]]
                    first_cell = False
                cell_dict[current_dict][(cv-cell_base_coords[0],cw-cell_base_coords[1],cu-cell_base_coords[2])] = [val, []]

def calcTransitions(centroid, stepped_centroid, coord, d=0, target_coord=[], mass=1.0):

    if len(target_coord) == len(coord):
        return [(mass, target_coord)]

    diff = stepped_centroid[d] - centroid[d]
    cell_lo = coord[d] + int(diff / cell_widths[d])
    cell_hi = cell_lo + 1
    prop_lo = 0.0
    if diff < 0.0: # actually, diff is negative so cell_lo is the upper cell
        cell_hi = cell_lo - 1
        prop_lo = ((diff % cell_widths[d]) / cell_widths[d])
    else:
        prop_lo = 1.0 - ((diff % cell_widths[d]) / cell_widths[d])
    prop_hi = 1.0 - prop_lo
    
    return calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_lo], mass*prop_lo) + calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_hi], mass*prop_hi)

for coord in cell_dict[current_dict]:
    centroid = [0 for a in range(dims)]

    for d in range(dims):
        centroid[d] = cell_base[d] + ((coord[d]+0.5)*cell_widths[d])

    stepped_centroid = cond(centroid)

    cell_dict[current_dict][coord][1] = calcTransitions(centroid, stepped_centroid, coord)

    
w_kernel_dim = 1
w_kernel_transitions = {}
cs = tuple([0 for d in range(dims)])
w_kernel_transitions[cs] = []
for c in range(len(pymiind_wI)):
    if pymiind_wI[c] > 0.0:
        w_kernel_transitions[cs] = w_kernel_transitions[cs] + [(pymiind_wI[c], [c-int(len(pymiind_wI)/2) if d == w_kernel_dim else 0 for d in range(dims)])]
w_kernel_transitions[cs] = [1.0,w_kernel_transitions[cs]]

u_kernel_dim = 2
u_kernel_transitions = {}
cs = tuple([0 for d in range(dims)])
u_kernel_transitions[cs] = []
for c in range(len(pymiind_uI)):
    if pymiind_uI[c] > 0.0:
        u_kernel_transitions[cs] = u_kernel_transitions[cs] + [(pymiind_uI[c], [c-int(len(pymiind_uI)/2) if d == u_kernel_dim else 0 for d in range(dims)])]
u_kernel_transitions[cs] = [1.0,u_kernel_transitions[cs]]

threshold = -50.4
reset = -70.6
threshold_reset_dim = 0
miind_threshold_cell = int((threshold - cell_base[threshold_reset_dim])/cell_widths[threshold_reset_dim])
miind_reset_cell = int((reset - cell_base[threshold_reset_dim])/cell_widths[threshold_reset_dim])

def updateCell(relative, new_cell_dict, transition, coord, mass, func):
    t = [a for a in transition[1]]
    if relative:
        for d in range(len(coord)):
            t[d] = coord[d] + t[d]

    if tuple(t) not in new_cell_dict.keys():
        new_cell_dict[tuple(t)] = [0.0,[]]
        centroid = [0 for a in range(len(coord))]

        for d in range(len(coord)):
            centroid[d] = cell_base[d] + ((t[d]+0.5)*cell_widths[d])

        stepped_centroid = func(centroid)
        new_cell_dict[tuple(t)][1] = calcTransitions(centroid, stepped_centroid, t)
    
    new_cell_dict[tuple(t)][0] += mass*transition[0]

def calcCellCentroid(coords):
    centroid = [0 for a in range(dims)]

    for d in range(dims):
        centroid[d] = cell_base[d] + ((coords[d]+0.5)*cell_widths[d])

    return centroid

def calcMarginals(cell_dict):
    vs = [{} for d in range(dims)]
    for c in cell_dict:
        for d in range(dims):
            if c[d] not in vs[d]:
                vs[d][c[d]] = cell_dict[c][0]
            else:
                vs[d][c[d]] += cell_dict[c][0]

    final_vs = [[] for d in range(dims)]
    final_vals = [[] for d in range(dims)]

    for d in range(dims):
        for v in vs[d]:
            final_vs[d] = final_vs[d] + [cell_base[d] + (cell_widths[d]*(v))]
            final_vals[d] = final_vals[d] + [vs[d][v]]

    return final_vs, final_vals

    
    #if CPUMIIND:

    epsilon = 0.0000001

    for a in cell_dict[(current_dict+1)%2].keys():
        cell_dict[(current_dict+1)%2][a][0] = 0.0
    
    for coord in cell_dict[current_dict]:
        cell_dict[current_dict][coord][0] /= remove_sum
        for ts in cell_dict[current_dict][coord][1]:
            updateCell(False, cell_dict[(current_dict+1)%2], ts, coord, cell_dict[current_dict][coord][0], cond)

    remove = []
    remove_sum = 1.0
    for a in cell_dict[(current_dict+1)%2].keys():
        if cell_dict[(current_dict+1)%2][a][0] < epsilon:
            remove = remove + [a]
            remove_sum -= cell_dict[(current_dict+1)%2][a][0]

    for a in remove:
        cell_dict[(current_dict+1)%2].pop(a, None)

    current_dict = (current_dict+1)%2

    for a in cell_dict[(current_dict+1)%2].keys():
        cell_dict[(current_dict+1)%2][a][0] = 0.0
    
    for coord in cell_dict[current_dict]:
        cell_dict[current_dict][coord][0] /= remove_sum
        for ts in w_kernel_transitions[(0,0,0)][1]:
            updateCell(True, cell_dict[(current_dict+1)%2], ts, coord, cell_dict[current_dict][coord][0], cond)

    remove = []
    remove_sum = 1.0
    for a in cell_dict[(current_dict+1)%2].keys():
        if cell_dict[(current_dict+1)%2][a][0] < epsilon:
            remove = remove + [a]
            remove_sum -= cell_dict[(current_dict+1)%2][a][0]

    for a in remove:
        cell_dict[(current_dict+1)%2].pop(a, None)

    current_dict = (current_dict+1)%2

    for a in cell_dict[(current_dict+1)%2].keys():
        cell_dict[(current_dict+1)%2][a][0] = 0.0
    
    for coord in cell_dict[current_dict]:
        cell_dict[current_dict][coord][0] /= remove_sum
        for ts in u_kernel_transitions[(0,0,0)][1]:
            updateCell(True, cell_dict[(current_dict+1)%2], ts, coord, cell_dict[current_dict][coord][0], cond)

    remove = []
    remove_sum = 1.0
    for a in cell_dict[(current_dict+1)%2].keys():
        if cell_dict[(current_dict+1)%2][a][0] < epsilon:
            remove = remove + [a]
            remove_sum -= cell_dict[(current_dict+1)%2][a][0]

    for a in remove:
        cell_dict[(current_dict+1)%2].pop(a, None)

    current_dict = (current_dict+1)%2
    
    print(len(cell_dict[current_dict].keys()))


    #if pyMIIND:

    cuda_function_applyJointTransition((v_res*w_res*u_res,),(128,),(v_res*w_res*u_res, pymiind_grid_2.data, cond_cells, cond_props, cond_counts, cond_offsets, pymiind_grid_1.data))
    cuda_function_convolveKernel((v_res*w_res*u_res,),(128,), (v_res*w_res*u_res, pymiind_grid_1.data, pymiind_grid_2.data, excitatory_kernel.data, wI_res, v_res))
    cuda_function_convolveKernel((v_res*w_res*u_res,),(128,), (v_res*w_res*u_res, pymiind_grid_2.data, pymiind_grid_1.data, inhibitory_kernel.data, uI_res, v_res*w_res))
    cp.copyto(pymiind_grid_1.data, pymiind_grid_2.data)
    

    if show_monte_carlo:
        # Also run the monte carlo simulation 
    
        fired_count = 0
        for nn in range(len(mc_neurons)):
            mc_neurons[nn] = cond(mc_neurons[nn])

            if (mc_neurons[nn][0] > -50.0):
                mc_neurons[nn][0] = -70.6
                fired_count+=1
                
            mc_neurons[nn][1] += epsp*poisson.rvs(w_rate*0.1) # override w
            mc_neurons[nn][2] += ipsp*poisson.rvs(u_rate*0.1) # override u

        monte_carlo_rates = monte_carlo_rates + [(fired_count / len(mc_neurons)) / 0.0001]

    if show_miind_redux:
        # Also run the MIIND (Redux) simulation
        cj.postRate(pop3, miind_ex, w_rate)
        cj.postRate(pop3, miind_in, u_rate)

        cj.step()
        #rates1 = rates1 + [cj.readRates()[0]*1000]
        mass = cj.readMass(pop3)
        miind_redux_rates = miind_redux_rates + [cj.readRates()[0]*1000]
    
        cj.update(miind_mass_grid, [a for a in mass])

        cj.marginal(miind_mass_grid, 2, miind_marginal_vw)
        cj.marginal(miind_mass_grid, 1, miind_marginal_vu)
        cj.marginal(miind_marginal_vw, 1, miind_marginal_v)
        cj.marginal(miind_marginal_vw, 0, miind_marginal_w)
        cj.marginal(miind_marginal_vu, 0, miind_marginal_u)

        miind_dist_v = cj.readDist(miind_marginal_v)
        miind_dist_w = cj.readDist(miind_marginal_w)
        miind_dist_u = cj.readDist(miind_marginal_u)

        miind_dist_v = [a / ((v_max-v_min)/v_res) for a in miind_dist_v]
        miind_dist_w = [a / ((w_max-w_min)/w_res) for a in miind_dist_w]
        miind_dist_u = [a / ((u_max-u_min)/u_res) for a in miind_dist_u]

    # The monte carlo hist function gives density not mass (booo)
    # so let's just convert to density here
    
    if (iteration % 50 == 0) :
        dist_v = cj.readDist(v0)
        dist_w = cj.readDist(w0)
        dist_u = cj.readDist(u0)

        dist_v = [a / ((v_max-v_min)/v_res) for a in dist_v]
        dist_w = [a / ((w_max-w_min)/w_res) for a in dist_w]
        dist_u = [a / ((u_max-u_min)/u_res) for a in dist_u]

        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(v, dist_v)
        ax[0,1].plot(w, dist_w)
        ax[1,0].plot(u, dist_u)
        if show_monte_carlo:
            ax[0,0].hist(mc_neurons[:,0], density=True, bins=v_res, range=[v_min,v_max], histtype='step')
            ax[0,1].hist(mc_neurons[:,1], density=True, bins=w_res, range=[w_min,w_max], histtype='step')
            ax[1,0].hist(mc_neurons[:,2], density=True, bins=u_res, range=[u_min,u_max], histtype='step')
        if show_miind_redux:
            ax[0,0].plot(np.linspace(v_min,v_max,v_res), miind_dist_v, linestyle='--')
            ax[0,1].plot(np.linspace(w_min,w_max,w_res), miind_dist_w, linestyle='--')
            ax[1,0].plot(np.linspace(u_min,u_max,u_res), miind_dist_u, linestyle='--')
        #if pyMIIND:
        py_miind_v = cp.asnumpy(cp.sum(pymiind_grid_2.data, (1,2)))
        py_miind_w = cp.asnumpy(cp.sum(pymiind_grid_2.data, (0,2)))
        py_miind_u = cp.asnumpy(cp.sum(pymiind_grid_2.data, (0,1)))

        py_miind_v = [a / ((v_max-v_min)/v_res) for a in py_miind_v]
        py_miind_w = [a / ((w_max-w_min)/w_res) for a in py_miind_w]
        py_miind_u = [a / ((u_max-u_min)/u_res) for a in py_miind_u]

        ax[0,0].plot(np.linspace(v_min,v_max,v_res), py_miind_v, linestyle='-.')
        ax[0,1].plot(np.linspace(w_min,w_max,w_res), py_miind_w, linestyle='-.')
        ax[1,0].plot(np.linspace(u_min,u_max,u_res), py_miind_u, linestyle='-.')
        #if CPUMIIND:
        mpos, marginals = calcMarginals(cell_dict[current_dict])

        marginals[0] = [a / (cell_widths[0]) for a in marginals[0]]
        marginals[1] = [a / (cell_widths[1]) for a in marginals[1]]
        marginals[2] = [a / (cell_widths[2]) for a in marginals[2]]

        ax[0,0].scatter(mpos[0], marginals[0])
        ax[0,1].scatter(mpos[1], marginals[1])
        ax[1,0].scatter(mpos[2], marginals[2])

        fig.tight_layout()
        plt.show()

    # Transfer v1,w1,u1 -> v0,w0,u0, transfer v2,w2,u2 -> v1,w1,u1
    cj.transfer(v1,v0)
    cj.transfer(w1,w0)
    cj.transfer(u1,u0)
    cj.transfer(v2,v1)
    cj.transfer(w2,w1)
    cj.transfer(u2,u1)
    cj.transfer(v2_w2, v1_w1)
    cj.transfer(v2_u2, v1_u1)

fig, ax = plt.subplots(1,1)
ax.plot(range(len(jazz_rates)), jazz_rates)
if show_monte_carlo:
    ax.plot(range(len(monte_carlo_rates)-2), monte_carlo_rates[2:])
if show_miind_redux:
    ax.plot(range(len(miind_redux_rates)), miind_redux_rates)
    
fig.tight_layout()
plt.show()

# Shutdown MIIND redux simulation
cj.shutdown()