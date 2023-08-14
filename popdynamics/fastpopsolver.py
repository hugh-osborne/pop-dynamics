import numpy as np
import cupy as cp

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import lil_matrix

class NdGrid:
    def __init__(self, _base, _size, _res, _data=None):
        self.base = [a for a in _base]
        self.size = [a for a in _size]
        self.res = [a for a in _res]
        if _data is not None:
            self.data = cp.asarray(_data,dtype=cp.float32)
            self.data = cp.ravel(self.data, order='C')

        temp_res_offsets = [1]
        r = [a for a in self.res]
        r.reverse()
        self.res_offsets = self.calcResOffsets(1, temp_res_offsets, r)
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

    def calcTransitions(self, centroid, stepped_centroid, coord, d=0, target_coord=[], mass=1.0):
        if len(target_coord) == len(coord):
            return [(mass, target_coord)]

        diff = stepped_centroid[d] - centroid[d]
        cell_lo = coord[d] + int(abs(diff) / self.cell_widths[d])
        cell_hi = cell_lo + 1
        prop_lo = 0.0
        if diff < 0.0: # actually, diff is negative so cell_lo is the upper cell
            cell_hi = cell_lo - 1
            prop_lo = ((abs(diff) % self.cell_widths[d]) / self.cell_widths[d])
        else:
            prop_lo = 1.0 - ((abs(diff) % self.cell_widths[d]) / self.cell_widths[d])
        prop_hi = 1.0 - prop_lo
    
        return self.calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_lo], mass*prop_lo) + self.calcTransitions(centroid, stepped_centroid, coord, d+1, target_coord + [cell_hi], mass*prop_hi)

class FastSolver:
    def __init__(self, _func, initial_distribution, _base, _size, _res):
        self.grids = [NdGrid(_base, _size, _res, initial_distribution),NdGrid(_base, _size, _res, initial_distribution)]
        self.current_grid = 0
        self.noise_kernels = []
        self.func = _func
        
        self.csr = self.generateConditionalTransitionCSR(self.grids[0], _func, self.grids[1])

    def generateConditionalTransitionCSR(self, grid_in, func, grid_out):
        lil_mat = lil_matrix((grid_in.total_cells,grid_out.total_cells))
        for r in range(grid_in.total_cells):
            start_point = grid_in.getCellCentroid(r)
            ts = grid_in.calcTransitions(start_point, func(start_point), grid_in.getCellCoords(r))
            for t in ts:
                out_cell = grid_out.getCellNum(t[1])
                if out_cell < 0:
                    out_cell = 0
                if out_cell >= grid_out.total_cells:
                    out_cell = grid_out.total_cells - 1
                lil_mat[r,out_cell] = t[0]

        return cp_csr_matrix(lil_mat)

    # Currently, just allow 1D arrays for noise and pair it with a dimension. 
    # Later we should allow the definition of ND kernels.
    def addNoiseKernel(self, kernel_data, dimension):
        self.noise_kernels = self.noise_kernels + [(dimension, cp.asarray(kernel_data, dtype=cp.float32))]

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
            dim_order = [i for i in range(self.grids[self.current_grid].numDimensions()) if i != kernel[0]]
            dim_order = dim_order + [kernel[0]]
            inv_order = [a for a in range(self.grids[self.current_grid].numDimensions())]
            d_rep = 0
            for d in range(self.grids[self.current_grid].numDimensions()):
                if d == kernel[0]:
                    inv_order[d] = self.grids[self.current_grid].numDimensions()-1
                else:
                    inv_order[d] = d_rep
                    d_rep += 1

            # Transpose the grid to make the contiguous dimension the same as the desired kernel dimension.
            transposed_grid = self.grids[self.current_grid].getTransposed(dim_order)
            transposed_res = [self.grids[self.current_grid].res[d] for d in dim_order]
            # Apply the convolution.
            transposed_grid = cp.convolve(transposed_grid, kernel[1], mode='same')
            # Transpose back to the original dimension order.
            transposed_grid = cp.ravel(cp.transpose(cp.reshape(transposed_grid, transposed_res , order='C'), inv_order), order='C')
            # Update the next grid.
            self.grids[(self.current_grid+1)%2].updateData(transposed_grid)
            self.current_grid = (self.current_grid+1)%2
