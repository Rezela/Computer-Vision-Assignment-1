import math
from collections import Counter, defaultdict
import pandas as pd

# 数据集
data = [
    [1, 'Sunny', 'High', 'High', 'No'],
    [2, 'Overcast', 'High', 'High', 'Yes'],
    [3, 'Rain', 'Medium', 'High', 'Yes'],
    [4, 'Rain', 'Low', 'Medium', 'Yes'],
    [5, 'Overcast', 'Low', 'Medium', 'Yes'],
    [6, 'Sunny', 'Medium', 'High', 'No'],
    [7, 'Sunny', 'Medium', 'Medium', 'Yes'],
    [8, 'Rain', 'Medium', 'Medium', 'Yes'],
    [9, 'Overcast', 'Medium', 'High', 'Yes']
]

df = pd.DataFrame(data, columns=['ID', 'Weather', 'Temperature', 'Wind', 'Class'])


# 计算熵
def entropy(class_list):
    total = len(class_list)
    counts = Counter(class_list)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


# 计算信息增益
def info_gain(df, attr, target='Class'):
    total_entropy = entropy(df[target])
    values = df[attr].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[attr] == v]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy


# ID3算法
def id3(df, target='Class', attributes=None, depth=0):
    indent = "  " * depth
    classes = Counter(df[target])

    # 如果纯净，返回类别
    if len(classes) == 1:
        return list(classes.keys())[0]

    if attributes is None:
        attributes = [col for col in df.columns if col not in [target, 'ID']]

    # 当前熵
    current_entropy = entropy(df[target])
    print(f"\n{indent}Tree node (Depth {depth})")
    print(f"{indent}Current Entropy: {current_entropy:.4f}")

    # 计算每个属性的信息增益
    gains = {attr: info_gain(df, attr, target) for attr in attributes}
    for attr, g in gains.items():
        print(f"{indent}Information Gain for - {attr}: {g:.4f}")

    # 选择最佳属性
    best_attr = max(gains, key=gains.get)
    print(f"{indent}Choose: {best_attr}")

    tree = {best_attr: {}}
    for v in df[best_attr].unique():
        subset = df[df[best_attr] == v]
        branch_entropy = entropy(subset[target])
        branch_counts = dict(Counter(subset[target]))
        print(f"{indent}  Branch {best_attr}={v}: distribution {branch_counts}, entropy={branch_entropy:.4f}")

        if len(subset) == 0:
            tree[best_attr][v] = classes.most_common(1)[0][0]
        else:
            new_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][v] = id3(subset, target, new_attrs, depth + 1)
    return tree


# 运行
decision_tree = id3(df)
print("\nFinal Decision Tree:")
print(decision_tree)
