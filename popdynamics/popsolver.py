import numpy as np
from .visualiser import Visualiser

class Solver:
    def __init__(self, _func, initial_distribution, _base, _cell_widths, _mass_epsilon, _vis=None):
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

    def draw(self):
        max_coords = self.coord_extent
        min_coords = self.coord_extent
        for a in self.cell_buffers[self.current_buffer].keys():
            max_coords = (max(max_coords[0],a[0]),max(max_coords[1],a[1]),max(max_coords[2],a[2]))
            min_coords = (min(min_coords[0],a[0]),min(min_coords[1],a[1]),min(min_coords[2],a[2]))
        self.coord_extent = (max_coords[0]-min_coords[0]+1, max_coords[1]-min_coords[1]+1, max_coords[2]-min_coords[2]+1)

        for a in self.cell_buffers[self.current_buffer].keys():
            self.visualiser.drawCell(a, self.cell_buffers[self.current_buffer][a][0], origin_location=(0.0,0.0,0.0), max_size=(2.0,2.0,2.0), max_res=self.coord_extent)
