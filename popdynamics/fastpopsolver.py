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
import cupy as cp

class NdGrid:
    def __init__(self, _base, _size, _res, _data=None):
        self.base = _base
        self.size = _size
        self.res = _res
        if _data is not None:
            self.data = cp.asarray(_data,dtype=cp.float32)

        temp_res_offsets = [1]
        self.res_offsets = self.calcResOffsets(1, temp_res_offsets, self.res)
        
        self.cell_widths = [self.size[a] / self.res[a] for a in range(self.numDimensions())]
        
        self.total_cells = 1
        for r in self.res:
            self.total_cells *= r

    def readData(self):
        return cp.asnumpy(self.data)

    def updateData(self, _data):
        self.data = cp.asarray(_data,dtype=cp.float32)

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

        i = self.numDimensions()-1
        while i >= 0:
            coords[i] = int(cell_num / self.res_offsets[i])
            cell_num = cell_num - (coords[i] * self.res_offsets[i])
            i -= 1

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

class GpuWrapper:
    def __init__(self):
        self.cuda_source = r'''
        extern "C"{

        __device__ int modulo(int a, int b) {
            int r = a % b;
            return r < 0 ? r + b : r;
        }

        __global__ void convolveKernel(
            unsigned int num_cells,
            float* grid_out,
            float* grid_in,
            float* kernel,
            unsigned int kernel_width,
            unsigned int dim_stride)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = index; i < num_cells; i += stride) {
                grid_out[i] = 0.0;
                for (int j = 0; j < kernel_width; j++) {
                    grid_out[i] += grid_in[modulo(i - ((j - int(kernel_width/2.0)) * dim_stride), num_cells)] * kernel[j];
                }
            }
        }

        __global__
        void applyJointTransition(
            unsigned int num_out_grid_cells,
            float* out_grid,
            unsigned int* transition_cells,
            float* transition_props,
            unsigned int* transitions_counts,
            unsigned int* transitions_offsets,
            float* in_grid) 
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;   
    
            for (int i = index; i < num_out_grid_cells; i += stride) {
                out_grid[i] = 0.0;
                for (int t = transitions_offsets[i]; t < transitions_offsets[i] + transitions_counts[i]; t++) {
                    out_grid[i] += in_grid[transition_cells[t]] * transition_props[t];
                }
            }
        }

        }'''

        self.cuda_module = cp.RawModule(code=self.cuda_source)
        self.cuda_function_applyJointTransition = self.cuda_module.get_function('applyJointTransition')
        self.cuda_function_convolveKernel = self.cuda_module.get_function('convolveKernel')

class FastSolver:
    def __init__(self, _func, initial_distribution, _base, _size, _res):
        self.grids = [NdGrid(_base, _size, _res, initial_distribution),NdGrid(_base, _size, _res, initial_distribution)]
        self.noise_kernels = []
        self.func = _func

        self.transition_data = self.generateConditionalTransitionCSR(self.grids[0], _func, self.grids[1])

        self.gpu_worker = GpuWrapper()

        
    def generateConditionalTransitionCSR(self, grid_in, func, grid_out):
        transitions = [[] for a in range(grid_out.total_cells)]
    
        offset = 0
        num_transitions = 0
        for r in range(grid_in.total_cells):
            start_point = grid_in.getCellCentroid(r)
            coords,weights = grid_out.getContainingCellWeightedCoords(func(start_point))
            transitions[grid_out.getCellNum(coords)] = transitions[grid_out.getCellNum(coords)] + [(r,1.0)]
            num_transitions += 1

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

    def addNoiseKernel(self, kernel, dimension):
        kernel_transitions = {}
        cs = tuple(np.zeros(self.dims).tolist())
        kernel_transitions[cs] = []
        for c in range(len(kernel)):
            if kernel[c] > 0.0:
                kernel_transitions[cs] = kernel_transitions[cs] + [(kernel[c], [c-int(len(kernel)/2) if d == dimension else 0 for d in range(self.dims)])]
        self.noise_kernels = self.noise_kernels + [[1.0,kernel_transitions[cs]]]

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

cuda_function_applyJointTransition((v_res*w_res*u_res,),(128,),(v_res*w_res*u_res, pymiind_grid_2.data, cond_cells, cond_props, cond_counts, cond_offsets, pymiind_grid_1.data))
cuda_function_convolveKernel((v_res*w_res*u_res,),(128,), (v_res*w_res*u_res, pymiind_grid_1.data, pymiind_grid_2.data, excitatory_kernel.data, I_res, v_res))
cuda_function_convolveKernel((v_res*w_res*u_res,),(128,), (v_res*w_res*u_res, pymiind_grid_2.data, pymiind_grid_1.data, inhibitory_kernel.data, I_res, v_res*w_res))
cp.copyto(pymiind_grid_1.data, pymiind_grid_2.data)
    
