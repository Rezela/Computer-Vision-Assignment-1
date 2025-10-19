import numpy as np
import pandas

data = [
    [2.0, 3.0, 1.0],
    [1.5, 2.5, 1.2],
    [8.0, 7.0, 9.0],
    [7.5, 6.5, 8.5],
    [2.5, 3.5, 1.5],
    [7.0, 6.0, 8.0],
    [1.8, 2.8, 1.1],
    [8.5, 7.5, 9.5]
]

columns = ['x', 'y', 'z']
df = pandas.DataFrame(data, columns=columns)

point1 = df.iloc[0].values
point2 = df.iloc[2].values

for i in range(2):
    print("Iteration: ", i+1)
    clusters = {0: [], 1: []}
    for j in range(len(df)):
        p = df.iloc[j].values
        distance1 = np.linalg.norm(p-point1)
        distance2 = np.linalg.norm(p-point2)
        # clusters[0 if distance1 < distance2 else 1].append(df.iloc[j])  # 保存点的数组到对应的簇
        clusters[0 if distance1 < distance2 else 1].append(j+1)  # 保存点的索引到对应的簇
    print("cluster[0]: ",clusters[0])
    print("cluster[1]: ",clusters[1])
    point1 = np.mean(df.iloc[[i-1 for i in clusters[0]]].values, axis=0)
    point2 = np.mean(df.iloc[[i-1 for i in clusters[1]]].values, axis=0)
    print("centroid 1: ",  point1,
          "\ncentroid 2: ", point2)
