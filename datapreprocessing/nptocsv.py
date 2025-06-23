import numpy as np
import pandas as pd

# Load the .npy file
X_test_pca = np.load("datapreprocessing\X_test_pca.npy")

# Convert to DataFrame
df = pd.DataFrame(X_test_pca)

# Save as CSV
df.to_csv("X_test_pca.csv", index=False)

print("Saved as X_test_pca.csv")