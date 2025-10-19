# Computing L1 and Cosine distances for KNN classification (K=3)
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

# Define dataset
data = [
    [1, 1.0, 2.0, 3.0, 'A'],
    [2, 0.5, 1.8, 2.7, 'B'],
    [3, 1.2, 2.2, 3.5, 'B'],
    [4, 4.6, 5.6, 3.7, 'A'],
    [5, 2.4, 4.6, 3.6, 'A'],
    [6, 3.5, 2.0, 4.1, 'B'],
    [7, 3.6, 4.6, 7.1, 'A'],
    [8, 6.2, 4.1, 1.3, 'B'],
    [9, 8.4, 3.5, 1.8, 'A'],
    [10, 5.8, 3.4, 2.7, 'B']
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['ID', 'x', 'y', 'z', 'Label'])

# Target point
target = np.array([3.5, 4.0, 6.0])

# Compute L1 distances
df['L1_Distance'] = df[['x', 'y', 'z']].apply(lambda row: np.sum(np.abs(row - target)), axis=1)

# Compute Cosine distances
features = df[['x', 'y', 'z']].values
cos_dist = cosine_distances(features, target.reshape(1, -1)).flatten()
df['Cosine_Distance'] = cos_dist

# Sort and get top 3 neighbors for each metric
top3_l1 = df.nsmallest(3, 'L1_Distance')
top3_cosine = df.nsmallest(3, 'Cosine_Distance')

# Determine classification by majority vote
l1_result = top3_l1['Label'].mode()[0]
cosine_result = top3_cosine['Label'].mode()[0]

# Output results
print("Full Distance Table:")
print(df[['ID', 'x', 'y', 'z', 'Label', 'L1_Distance', 'Cosine_Distance']])

print("\nTop 3 Neighbors by L1 Distance:")
print(top3_l1[['ID', 'Label', 'L1_Distance']])

print("\nTop 3 Neighbors by Cosine Distance:")
print(top3_cosine[['ID', 'Label', 'Cosine_Distance']])

print(f"\nFinal Classification by L1 Distance: {l1_result}")
print(f"Final Classification by Cosine Distance: {cosine_result}")
