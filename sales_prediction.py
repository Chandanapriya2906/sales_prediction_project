import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# Load Dataset
# ==============================
data = pd.read_csv("Advertising.csv", sep="\t")

# Remove unwanted column
data = data.drop("Unnamed: 0", axis=1)

# ==============================
# Features and Target
# ==============================
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# ==============================
# Split Dataset
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train Model
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# Save Trained Model
# ==============================
joblib.dump(model, "sales_prediction_model.pkl")
print("✅ Model saved successfully!")

# ==============================
# Predict Test Data
# ==============================
y_pred = model.predict(X_test)

# ==============================
# Model Evaluation
# ==============================
print("\nModel Performance")
print("--------------------------")
print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE :", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R² Score:", round(r2_score(y_test, y_pred), 4))

# ==============================
# User Prediction
# ==============================
print("\nEnter Advertising Budgets")

tv = float(input("TV Budget: "))
radio = float(input("Radio Budget: "))
newspaper = float(input("Newspaper Budget: "))

new_data = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [newspaper]
})

prediction = model.predict(new_data)

print("\n📈 Predicted Sales =", round(prediction[0], 2))

# ==============================
# Actual vs Predicted Graph
# ==============================
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()