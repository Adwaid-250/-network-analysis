import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import joblib

# Load your saved features
X_train = pd.read_csv('X_trainscaled.csv')
X_test = pd.read_csv('X_testscaled.csv')

# Step 1: Fit PCA on the training set
n_components = 4
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

# Step 2: Apply the same transformation on the test set
X_test_pca = pca.transform(X_test)
joblib.dump(pca, 'pca_model.pkl')
# # Step 3 (Optional): Save reduced data
# np.save('X_train_pca.npy', X_train_pca)
# np.save('X_test_pca.npy', X_test_pca)

# # Step 4 (Optional): Check explained variance
# print("Explained variance per component:", pca.explained_variance_ratio_)
# print("Total variance preserved:", sum(pca.explained_variance_ratio_))

# Save the reduced features as CSV
pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n_components)]).to_csv("X_train_pca.csv", index=False)

print("Saved reduced features to X_train_pca.csv")

