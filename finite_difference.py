import numpy as np
from itertools import product

class DiscreteGrid:
    def __init__(self, sizes, min_v, max_v):
        self.sizes = np.asarray(sizes)
        self.N = np.prod(self.sizes)
        self.pref_size = np.cumprod([1] + sizes[:-1])
        self.min_v = np.asarray(min_v)
        self.max_v = np.asarray(max_v)
        self.hs = (self.max_v - self.min_v) / (self.sizes - 1)

    def get_idx(self, xs):
        return np.sum(np.asarray(xs) * self.pref_size)

    def pos_inside(self, xs):
        xs = np.asarray(xs)
        return np.all(0 <= xs) and np.all(xs < self.sizes) 

    def get_pos(self, idx):
        xs = np.zeros((len(self.sizes)))
        p = np.cumprod([1] + self.sizes)
        for i in range(len(self.sizes)):
            xs[i] = (idx % p[i + 1]) // p[i]
        return xs

    def discretize_function(self, f):
        f_ = np.zeros(self.N)

        for xs in product(*[range(n) for n in self.sizes]):
            # print(xs)
            # print(self.get_idx(xs))
            # print('point: ', np.asarray(xs) / self.sizes * (self.max_v - self.min_v) + self.min_v)
            f_[self.get_idx(xs)] = f(np.asarray(xs) / self.sizes * (self.max_v - self.min_v) + self.min_v)

        return f_

    def derivative(self, dxs):
        if len(dxs) != len(self.sizes):
            raise ValueError('Dimension of parameter is wrong.')

        A = np.zeros((self.N, self.N))
    
        dxs_ = []
        denominator = 1
        for i in range(len(self.sizes)):
            if dxs[i] == 0:
                dxs_.append([(0, 1)])
            elif dxs[i] == 1:
                dxs_.append([(-1, -1), (1, 1)])
                denominator *= self.hs[i] * 2
            elif dxs[i] == 2:
                dxs_.append([(-1, 1), (0, -2), (1, 1)])
                denominator *= self.hs[i]**2
            else:
                raise ValueError('Only 1st and 2nd derivatives allowed')
            
        dxs_ = np.array(list(product(*dxs_)))
        area = []
        for x in dxs_:
            area.append((x[:, 0], np.prod(x[:, 1])))

        for xs in product(*[range(n) for n in self.sizes]):
            xs = np.asarray(xs)
            i = self.get_idx(xs)
            for diff, val in area:
                cur_xs = xs + diff
                if self.pos_inside(cur_xs):
                    j = self.get_idx(cur_xs)
                    A[i, j] += val
        
        return A / denominator
