class SC_DataGenerator(object):
    def __init__(self):
        self.num_service = 0
        self.count = 0
        self.act_count =[]
    def init(self, f_path):
        # mase -- 打开了nodeSet.txt 文件
        f=open(f_path)
        # mase -- 存储的是 每个 state节点 候选服务的数量
        num_service = []
        # mase -- 这里有两次readline
        # mase -- 原因是，nodeSet文件中第一行为 "#"，所以第一个readline读的是 "#"
        # mase -- 第二个readline读的就是state节点，并且需要保存到 line 变量中
        f.readline()
        line = f.readline()
        # mase -- candidates_c保存的就是每个 state 节点，是一个数组形式
        candidates_c = line.split(' ')
        # mase -- 不是很理解， candidates 和 candidates_c 似乎是一模一样的数组
        candidates = []
        for index in range(len(candidates_c)):
            candidates.append(candidates_c[index])
        max = 0
        x = 0
        for candidate in candidates:
            num = 0
            count = 0
            # mase -- f1 存放的是 该 state 节点 对应的候选服务 txt文件
            f1 = open('服务名聚类最终结果/' + candidate + '.txt')
            line1 = f1.readline()
            while line1:
                num = num + 1
                line1 = f1.readline()
            num_service.append(num)
            # if num > max:
            #     max = num
            #     x = candidate
        act_count = [0]
        for i in num_service:
            count += i
            act_count.append(count)
        self.num_service = num_service
        self.count = count
        self.act_count = act_count
        # print(x)
    def get_num_service(self):
        return self.num_service
# num_service储存了每个子任务的候选服务数量

if __name__ == "__main__":
    pass
