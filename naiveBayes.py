# %%
# Import packages

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# %%
# Import data
training = pd.read_csv("WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv")

# %%
# Clean with data 


# %% 
# Create the X and Y
xtrain = training.drop(training.columns[[0,1,2,3,4,5]], axis=1)
ytrain = training.loc[:, 'num_window']

xtest = test.drop(test.columns[[0,1,2,3,4,5]], axis=1)
ytest = test.loc[:, 'num_window']

# %%
# Init the Gaussian Classifier
model = GaussianNB()
# %%
# Train the model
model.fit(xtrain, ytrain)
# %%
# Predict Output 
pred = model.predict(xtest)
print(pred[:5])
# %%
# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
# %%
