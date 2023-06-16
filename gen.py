import random
import csv
from pern46 import reward_func_pre as rp

def Get_numservice(f):
    num_service = []
    f.readline()
    line = f.readline()
    candidates_c = line.split(' ')
    candidates = []
    candidates = [candidates_c[index] for index in range(len(candidates_c))]
    # print('Candidates: ',candidates)
    for candidate in candidates:
        num = 0
        # rows = 0  # 使得服务限制在2个
        f1 = open('服务名聚类最终结果/' + candidate + '.txt')
        line1 = f1.readline()
        while line1:
            num = num + 1
            line1 = f1.readline()
        num_service.append(num)
    return num_service

#if __name__ =='__main__':
def generate_data(nodedata):
    path = "data/nodeSet.txt"
    f = open(path)
    num_service = Get_numservice(f)
    normResponseTime = (-QoS_Data['Response Time'] + maxResponseTime) / (maxResponseTime - minResponseTime)
    normAvailbility = (QoS_Data['Availability'] - minAvailbility) / (maxAvailbility - minAvailbility)
    normThroughput = (QoS_Data['Throughput'] - minThroughput) / (maxThroughput - minThroughput)

    def calculate_reward(position):
        return reward_func_pre(position, node_data, normResponseTime, normAvailbility, normThroughput)

    print(result)
    csvfile = open('pretrain/result.csv', "w",newline="")
    writer = csv.writer(csvfile)
    writer.writerows(result)
    csvfile.close()


