import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# Load scaled features and labels (not yet PCA-reduced)
X_train = pd.read_csv("datapreprocessing/X_trainscaled.csv")
y_train = pd.read_csv("datapreprocessing/y_train.csv").squeeze()
X_test = pd.read_csv("datapreprocessing/X_testscaled.csv")
y_test = pd.read_csv("datapreprocessing/y_test.csv").squeeze()

# Apply PCA (fit on train, transform both train and test)
n_components = 4  # Set to desired number of components/qubits
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert to DataFrame with consistent column names
columns = [f'PC{i+1}' for i in range(n_components)]
X_train_pca = pd.DataFrame(X_train_pca, columns=columns)
X_test_pca = pd.DataFrame(X_test_pca, columns=columns)

# Initialize CatBoostClassifier for multiclass
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    verbose=0,
    random_seed=42
)

# Train the model
model.fit(X_train_pca, y_train)

# Predict on test set
y_pred = model.predict(X_test_pca)
y_pred = y_pred.flatten()  # Ensure shape matches y_test

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained CatBoost model
model.save_model("catboost_qml_model.cbm")
print("Model saved as catboost_qml_model.cbm")