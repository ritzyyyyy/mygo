import numpy as np
class SumTree1(object):
    # data样本的指针，从0开始往后移动，同时也影响叶子节点的指针位置
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # 叶节点的个数，叶节点存放 priority 值
        self.tree = np.zeros(2 * capacity - 1)     # 树节点
        # [--------------父节点个数是 叶节点-1-------------][-------叶节点存放priority值，个数是capacity-------]
        #             size: capacity - 1                                size: capacity
        self.data = np.zeros(capacity, dtype=list)  # for all transitions
        # [--------------存放样本的 data 大小是capacity 和叶节点个数一样-------------]
        #                        size: capacity

    def add(self, p, data):
        # 存priority的值是从叶节点开始，叶节点的索引从capacity - 1开始
        tree_idx = self.data_pointer + self.capacity - 1  # 当前叶子节点的索引
        self.data[self.data_pointer] = data  # 更新data数组， 存入的data值是传过来的样本[s,a,r,s_]
        self.update(tree_idx, p)  # 由于添加了p值或更新了p值，所以需要更新树

        self.data_pointer += 1  # 更新data数组后，其指针往后移一位，同时也导致叶子节点的指针也往后移一位
        if self.data_pointer >= self.capacity:  # 当data容量满了之后，指针从头开始
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # 从树 下往上走
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # 触底，此时 parent_index 就在叶节点
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        # 因为有  tree_idx - (capacity - 1)= data_pointer  所以有了叶节点的指针后，可以用这个式子得到data数组中的指针
        data_idx = leaf_idx - self.capacity + 1
        # 返回  ①叶节点索引 ② 叶节点的p值 ③data数组中的对应的一个样本
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]  # the root