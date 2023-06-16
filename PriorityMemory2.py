from PrioritySumTree2 import SumTree2
import numpy as np
class Memory2(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree2(capacity)
        self.prio_max = 0.1
        self.epsilon = 0.01
        self.alpha = 0.6

    def store(self, transition):
        # 负索引切片是因为 从树的 倒数第8个索引开始截取，也就是只截取叶节点，然后从中选取最大的priority值
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        # add 方法做的事就是，存样本进data，添加新的p值到二叉树上，同时更新二叉树
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory = np.empty((n,), dtype=np.int32), np.empty((n, 4))
        pri_seg = self.tree.total() / n       # priority segment
        # self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()     # for later calculate ISweight
        # if min_prob == 0:  # 因为min_prob是分母，所以不能让他为0
        #     min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get(v)
            # prob = p / self.tree.total()
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory

    def batch_update(self, tree_idx, abs_errors):
        self.prio_max = max(self.prio_max, max(np.abs(abs_errors)))
        for i, idx in enumerate(tree_idx):
            p = (np.abs(abs_errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
        # abs_errors += self.epsilon  # convert to abs and avoid 0
        # clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        # ps = np.power(clipped_errors, self.alpha)
        # for ti, p in zip(tree_idx, ps):
        #     self.tree.update(ti, p)