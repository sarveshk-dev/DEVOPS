import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

data = fetch_california_housing()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Features Shape :", X.shape)
print("Target Shape   :", y.shape)

# -----------------------------------------------------
# Split Dataset
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Completed")

# -----------------------------------------------------
# Create Model
# -----------------------------------------------------

model = LinearRegression()

print("\nLinear Regression Model Created")

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed")

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

predictions = model.predict(X_test)

# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# -----------------------------------------------------
# Save Model
# -----------------------------------------------------

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")
print("Saved as model.pkl")
