# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("Advertising.csv", sep="\t")

# Remove unnecessary column if present
if "Unnamed: 0" in data.columns:
    data = data.drop("Unnamed: 0", axis=1)

# Features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model evaluation
print("Model Evaluation")
print("-----------------")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE :", mean_squared_error(y_test, y_pred)**0.5)
print("R2 Score :", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "sales_model.pkl")

print("\nModel saved successfully as sales_model.pkl")