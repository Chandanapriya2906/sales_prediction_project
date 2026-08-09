import joblib
import pandas as pd

# Load the saved model
model = joblib.load("sales_prediction_model.pkl")

# New input
new_data = pd.DataFrame({
    "TV": [230],
    "Radio": [38],
    "Newspaper": [69]
})

prediction = model.predict(new_data)

print("Predicted Sales:", round(prediction[0], 2))