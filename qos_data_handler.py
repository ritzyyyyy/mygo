import pandas as pd

QoS_Data = pd.read_csv('./data/QWS_Dataset_With_Head.csv')

maxResponseTime = QoS_Data.iloc[:, [1]].values.max()
maxAvailbility = QoS_Data.iloc[:, [2]].values.max()
maxThroughput = QoS_Data.iloc[:, [3]].values.max()
minResponseTime = QoS_Data.iloc[:, [1]].values.min()
minAvailbility = QoS_Data.iloc[:, [2]].values.min()
minThroughput = QoS_Data.iloc[:, [3]].values.min()

normResponseTime = (-QoS_Data['Response Time'] + maxResponseTime) / (maxResponseTime - minResponseTime)
normAvailbility = (QoS_Data['Availability'] - minAvailbility) / (maxAvailbility - minAvailbility)
normThroughput = (QoS_Data['Throughput'] - minThroughput) / (maxThroughput - minThroughput)

def get_normalized_qos_values(atom_node_id):
    return normResponseTime[atom_node_id - 1], normAvailbility[atom_node_id - 1], normThroughput[atom_node_id - 1]
