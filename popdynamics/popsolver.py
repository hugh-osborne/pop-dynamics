import numpy as np
from .visualiser import Visualiser

class Solver:
    def __init__(self, _func, initial_distribution, _base, _cell_widths, _mass_epsilon, _vis=None, vis_dimensions=(0,1,2)):
        self.dims = _base.shape[0]
        self.cell_widths = _cell_widths

        self.cell_buffers = [{},{}]

        self.cell_base = np.zeros(self.dims)
        self.current_buffer = 0
        self.remove_sum = 1.0

        self.noise_kernels = []

        self.mass_epsilon = _mass_epsilon
        self.func = _func

        self.visualiser = _vis
        self.coord_extent = np.ones(self.dims) # The number of cells in each dimension direction
        self.max_mass = 1.0
        self.vis_dimensions = vis_dimensions

        first_cell = True
        cell_base_coords = np.zeros(self.dims)
        for idx, val in np.ndenumerate(initial_distribution):
            if val > 0.0:
                if first_cell:
                    cell_base_coords = idx
                    self.cell_base = _base + (np.multiply(idx,_cell_widths))
                    first_cell = False
                self.cell_buffers[self.current_buffer][tuple((np.asarray(idx)-cell_base_coords).tolist())] = [val, []]

        # init first cell buffer

        for coord in self.cell_buffers[self.current_buffer]:
            centroid = [0 for a in range(self.dims)]

            for d in range(self.dims):
                centroid[d] = self.cell_base[d] + ((coord[d]+0.5)*self.cell_widths[d])

            stepped_centroid = self.func(centroid)

            self.cell_buffers[self.current_buffer][coord][1] = self.calcTransitions(centroid, stepped_centroid, coord)

    def addNoiseKernel(self, kernel, dimension):
        kernel_transitions = {}
        cs = tuple(np.zeros(self.dims).tolist())
        kernel_transitions[cs] = []
        for c in range(len(kernel)):
            if kernel[c] > 0.0:
                kernel_transitions[cs] = kernel_transitions[cs] + [(kernel[c], [c-int(len(kernel)/2) if d == dimension else 0 for d in range(self.dims)])]
        self.noise_kernels = self.noise_kernels + [[1.0,kernel_transitions[cs]]]

    def calcTransitions(self, centroid, stepped_centroid, coord, d=0, target_coord=[], mass=1.0):
        if len(target_coord) == len(coord):
            return [(mass, target_coord)]

        diff = stepped_centroid[d] - centroid[d]
        cell_lo = coord[d] + int(diff / self.cell_widths[d])
        cell_hi = cell_lo + 1
        prop_lo = 0.0
        if diff < 0.0: # actually, diff is negative so cell_lo is the upper cell
            cell_hi = cell_lo - 1
            prop_lo = ((diff % self.cell_widths[d]) / self.cell_widths[d])
        else:
            prop_lo = 1.0 - ((diff % self.cell_widths[d]) / self.cell_widths[d])
        prop_hi = 1.0 - prop_lo
    
        return self.calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_lo], mass*prop_lo) + self.calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_hi], mass*prop_hi)

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
            new_cell_dict[tuple(t)][1] = self.calcTransitions(centroid, stepped_centroid, t)
    
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

    def calcMarginal(self, dimensions):
        vals = {}
        for cell_key, cell_val in self.cell_buffers[self.current_buffer].items():
            reduced_key = tuple([cell_key[a] for a in range(self.dims) if a in dimensions])
            if reduced_key not in vals:
                vals[reduced_key] = cell_val[0]
            else:
                vals[reduced_key] += cell_val[0]

        final_centroids = [np.zeros(len([a for a in dimensions])) for a in vals]
        final_coords = [k for k in vals.keys()]
        final_vals = [v[1] for v in vals.items()]
        
        i = 0
        for v_key, v_val in vals.items():
            for d in range(len([a for a in dimensions])):
                final_centroids[i][d] = self.cell_base[dimensions[d]] + (self.cell_widths[dimensions[d]]*(v_key[d]))
            i += 1

        return final_coords, final_centroids, final_vals


    def updateDeterministic(self):
        # Set the next buffer mass values to 0
        for a in self.cell_buffers[(self.current_buffer+1)%2].keys():
            self.cell_buffers[(self.current_buffer+1)%2][a][0] = 0.0
    
        # Fill the next buffer with the updated mass
        for coord in self.cell_buffers[self.current_buffer]:
            self.cell_buffers[self.current_buffer][coord][0] /= self.remove_sum
            for ts in self.cell_buffers[self.current_buffer][coord][1]:
                self.updateCell(False, self.cell_buffers[(self.current_buffer+1)%2], ts, coord, self.cell_buffers[self.current_buffer][coord][0], self.func)

        # Remove any cells with a small amount of mass and keep a total to spread back to the remaining population
        remove = []
        self.remove_sum = 1.0
        for a in self.cell_buffers[(self.current_buffer+1)%2].keys():
            if self.cell_buffers[(self.current_buffer+1)%2][a][0] < self.mass_epsilon:
                remove = remove + [a]
                self.remove_sum -= self.cell_buffers[(self.current_buffer+1)%2][a][0]

        for a in remove:
            self.cell_buffers[(self.current_buffer+1)%2].pop(a, None)

        # swap the buffer counter
        self.current_buffer = (self.current_buffer+1)%2

    def applyNoiseKernels(self):
        for kernel in self.noise_kernels:
            for a in self.cell_buffers[(self.current_buffer+1)%2].keys():
                self.cell_buffers[(self.current_buffer+1)%2][a][0] = 0.0
    
            for coord in self.cell_buffers[self.current_buffer]:
                self.cell_buffers[self.current_buffer][coord][0] /= self.remove_sum
                for ts in kernel[1]:
                    self.updateCell(True, self.cell_buffers[(self.current_buffer+1)%2], ts, coord, self.cell_buffers[self.current_buffer][coord][0], self.func)

            remove = []
            self.remove_sum = 1.0
            for a in self.cell_buffers[(self.current_buffer+1)%2].keys():
                if self.cell_buffers[(self.current_buffer+1)%2][a][0] < self.mass_epsilon:
                    remove = remove + [a]
                    self.remove_sum -= self.cell_buffers[(self.current_buffer+1)%2][a][0]

            for a in remove:
                self.cell_buffers[(self.current_buffer+1)%2].pop(a, None)

            self.current_buffer = (self.current_buffer+1)%2

    def draw(self, grid_res_override=None):
        if not self.visualiser.beginRendering():
            return
        max_coords = tuple([1 for a in self.vis_dimensions])
        min_coords = tuple([1 for a in self.vis_dimensions])
        self.max_mass = 0.0

        mcoords, mcentroids, mvals = self.calcMarginal(self.vis_dimensions)
        for a in range(len(mvals)):
            max_coords = tuple([max(max_coords[i],mcoords[a][i]) for i in range(len(self.vis_dimensions))])
            min_coords = tuple([min(min_coords[i],mcoords[a][i]) for i in range(len(self.vis_dimensions))])
            self.max_mass = max(self.max_mass, mvals[a])
        self.coord_extent = tuple([max(10,max_coords[a]-min_coords[a]+1) for a in range(len(self.vis_dimensions))])
        
        if grid_res_override != None:
            self.coord_extent = grid_res_override

        for a in range(len(mvals)):
            if mvals[a] < 0.000001:
                continue
            self.visualiser.drawCell(mcoords[a], mvals[a] / self.max_mass, origin_location=tuple([0.0 for d in range(len(self.vis_dimensions))]), max_size=tuple([2.0 for d in range(len(self.vis_dimensions))]), max_res=self.coord_extent)

        self.visualiser.endRendering()
