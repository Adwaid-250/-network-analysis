import pandas as pd

# Load the CSV file
df = pd.read_csv("y_test.csv")

# Count occurrences of each number in the entire DataFrame
number_counts = pd.Series(df.values.ravel()).value_counts().sort_index()

print(number_counts)