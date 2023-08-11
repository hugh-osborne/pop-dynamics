import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import lil_matrix

class NdGrid:
    def __init__(self, _base, _size, _res, _data=None):
        self.base = _base
        self.size = _size
        self.res = _res
        if _data is not None:
            self.data = cp.asarray(_data,dtype=cp.float32)
            self.data = cp.ravel(self.data, order='C')

        temp_res_offsets = [1]
        self.res_offsets = self.calcResOffsets(1, temp_res_offsets, self.res)
        self.res_offsets.reverse()
        
        self.cell_widths = [self.size[a] / self.res[a] for a in range(self.numDimensions())]
        
        self.total_cells = 1
        for r in self.res:
            self.total_cells *= r

    def readData(self):
        return cp.asnumpy(cp.reshape(self.data, self.res, order='C'))

    def updateData(self, _data):
        self.data = cp.asarray(_data,dtype=cp.float32)
        self.data = cp.ravel(self.data, order='C')

    def getTransposed(self, _ord):
        return cp.ravel(cp.transpose(cp.reshape(self.data, self.res, order='C'), _ord), order='C')

    def calcResOffsets(self, count, offsets, res):
        if len(res) == 1:
            return offsets

        count *= res[0]
        offsets = offsets + [count]

        new_res = []
        for i in [1+a for a in range(len(res)-1)]:
            new_res = new_res + [res[i]]

        if len(new_res) == 1:
            return offsets

        return self.calcResOffsets(count, offsets, new_res)

    def numDimensions(self):
        return len(self.base)

    def getCellCoords(self, cell_num):
        coords = [0 for a in range(self.numDimensions())]

        for i in range(self.numDimensions()):
            coords[i] = int(cell_num / self.res_offsets[i])
            cell_num = cell_num - (coords[i] * self.res_offsets[i])

        return coords

    def getCellNum(self, coords):
        cell_num = 0
        for i in range(self.numDimensions()):
            cell_num += coords[i] * self.res_offsets[i]

        return cell_num

    def getCellCentroid(self, cell_num):
        coords = self.getCellCoords(cell_num)
        centroid = [0 for a in range(self.numDimensions())]

        for d in range(self.numDimensions()):
            centroid[d] = self.base[d] + ((coords[d]+0.5)*self.cell_widths[d])

        return centroid

    def getContainingCellWeightedCoords(self, point):
        coords = [0 for a in range(self.numDimensions())]
        weights = [1.0 for a in range(self.numDimensions())]

        for d in range(self.numDimensions()):
            exact = (point[d]-self.base[d]) / self.cell_widths[d]
            coords[d] = int(exact)
            weights[d] = exact - coords[d]
            if coords[d] < 0:
                coords[d] = 0
                weights[d] = 1.0
            if coords[d] >= self.res[d]:
                coords[d] = self.res[d]-1
                weights[d] = 1.0

        return coords, weights

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

class FastSolver:
    def __init__(self, _func, initial_distribution, _base, _size, _res):
        self.grids = [NdGrid(_base, _size, _res, initial_distribution),NdGrid(_base, _size, _res, initial_distribution)]
        self.current_grid = 0
        self.noise_kernels = []
        self.func = _func

        self.csr = cp_csr_matrix((self.grids[0].total_cells, self.grids[1].total_cells), dtype=cp.float32)

        transition_data = self.generateConditionalTransitionCSR(self.grids[0], _func, self.grids[1])
        self.transition_data = [cp.asarray(transition_data[0]),cp.asarray(transition_data[1],dtype=cp.float32), cp.asarray(transition_data[2]), cp.asarray(transition_data[3])]
        for a in self.transition_data:
            a = cp.reshape(a, a.shape, order='C')

    def generateConditionalTransitionCSR(self, grid_in, func, grid_out):
        transitions = [[] for a in range(grid_out.total_cells)]
    
        offset = 0
        num_transitions = 0
        lil_mat = lil_matrix((self.grids[self.current_grid].total_cells,self.grids[self.current_grid].total_cells))
        for r in range(grid_in.total_cells):
            start_point = grid_in.getCellCentroid(r)
            ts = grid_in.calcTransitions(start_point, func(start_point), grid_in.getCellCoords(r))
            for t in ts:
                out_cell = grid_out.getCellNum(t[1])
                if out_cell < 0:
                    out_cell = 0
                if out_cell >= grid_out.total_cells:
                    out_cell = grid_out.total_cells - 1
                lil_mat[out_cell,r] = t[0]
                transitions[out_cell] = transitions[out_cell] + [(r,t[0])]
                num_transitions += 1

        self.csr = cp_csr_matrix(lil_mat)

        out_transitions_cells = [a for a in range(num_transitions)]
        out_transitions_props = [a for a in range(num_transitions)]
        out_transitions_counts = [a for a in range(grid_out.total_cells)]
        out_transitions_offsets = [a for a in range(grid_out.total_cells)]

        transition_count = 0
        cell_count = 0
        for t in transitions:
            # Don't worry about weighting just yet just do a single cell
            count = len(t)
            for r in t:
                out_transitions_cells[transition_count] = r[0]
                out_transitions_props[transition_count] = r[1]
                transition_count += 1
            out_transitions_offsets[cell_count] = offset
            out_transitions_counts[cell_count] = count
            offset += count
            cell_count += 1

        return out_transitions_cells, out_transitions_props, out_transitions_counts, out_transitions_offsets 

    # Currently, just allow 1D arrays for noise and pair it with a dimension. 
    # Later we should allow the definition of ND kernels.
    def addNoiseKernel(self, _base, _size, _res, kernel_data, dimension):

        # store kernels as csrs
        lil_mat = lil_matrix((self.grids[self.current_grid].total_cells,self.grids[self.current_grid].total_cells))

        for r in range(self.grids[self.current_grid].total_cells):
            for k in range(len(kernel_data)):
                idx = r - int(len(kernel_data)/2) + k
                if idx < 0:
                    idx = 0
                if idx >= self.grids[self.current_grid].total_cells:
                    idx = self.grids[self.current_grid].total_cells - 1
                lil_mat[idx,r] = kernel_data[k]

        self.noise_kernels = self.noise_kernels + [cp_csr_matrix(lil_mat)]

    # Do CPU marginal calculation for now. Slow because we need to move the full distribution off card
    def calcMarginals(self):
        final_vals = []
        final_vs = []
        for d in range(self.grids[self.current_grid].numDimensions()):
            other_dims = tuple([i for i in range(self.grids[self.current_grid].numDimensions()) if i != d])
            final_vals = final_vals + [np.sum(self.grids[self.current_grid].readData(), other_dims)]
            final_vs = final_vs + [np.linspace(self.grids[self.current_grid].base[d],self.grids[self.current_grid].base[d] + self.grids[self.current_grid].size[d],self.grids[self.current_grid].res[d])]

        return final_vs, final_vals

    def updateDeterministic(self):
        self.grids[(self.current_grid+1)%2].updateData(self.csr.dot(self.grids[self.current_grid].data))
        self.current_grid = (self.current_grid+1)%2

    def applyNoiseKernels(self):
        for kernel in self.noise_kernels:
            self.grids[(self.current_grid+1)%2].updateData(kernel.dot(self.grids[self.current_grid].data))
            self.current_grid = (self.current_grid+1)%2
