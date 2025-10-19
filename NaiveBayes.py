# Implementing Naive Bayes classifier to predict Play Golf for a new sample

import pandas as pd
import numpy as np

# Define dataset
data = [
    ['Sunny', 'Hot', 'High', False, 'No'],
    ['Overcast', 'Hot', 'High', False, 'Yes'],
    ['Rain', 'Mild', 'High', False, 'Yes'],
    ['Sunny', 'Mild', 'Normal', False, 'Yes'],
    ['Sunny', 'Cool', 'Normal', True, 'No'],
    ['Overcast', 'Cool', 'Normal', True, 'Yes'],
    ['Rain', 'Mild', 'Normal', False, 'Yes'],
    ['Rain', 'Cool', 'Normal', True, 'No']
]

columns = ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play Golf']
df = pd.DataFrame(data, columns=columns)

# New sample
new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Hot',
    'Humidity': 'Normal',
    'Windy': False
}

# Step 1: Prior probabilities
total = len(df)
yes_count = len(df[df['Play Golf'] == 'Yes'])
no_count = len(df[df['Play Golf'] == 'No'])

P_yes = yes_count / total
P_no = no_count / total

# Step 2: Conditional probabilities
def conditional_prob(feature, value, label):
    subset = df[df['Play Golf'] == label]
    count = len(subset[subset[feature] == value])
    return count / len(subset)

features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
P_X_given_yes = np.prod([conditional_prob(f, new_sample[f], 'Yes') for f in features])
P_X_given_no = np.prod([conditional_prob(f, new_sample[f], 'No') for f in features])

# Step 3: Posterior probabilities
posterior_yes = P_X_given_yes * P_yes
posterior_no = P_X_given_no * P_no

# Step 4: Prediction
prediction = 'Yes' if posterior_yes > posterior_no else 'No'

# Print results
print("Prior P(Yes):", round(P_yes, 3))
print("Prior P(No):", round(P_no, 3))
print("Likelihood P(X|Yes):", round(P_X_given_yes, 5))
print("Likelihood P(X|No):", round(P_X_given_no, 5))
print("Posterior P(Yes|X):", round(posterior_yes, 5))
print("Posterior P(No|X):", round(posterior_no, 5))
print("Prediction for new sample:", prediction)
