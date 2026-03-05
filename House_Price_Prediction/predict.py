import joblib
import json
import pandas as pd

model = joblib.load('house_price_model.pkl')

with open("columns.json", "r") as f:
    model_columns = json.load(f)

area = float(input("Enter the area in square feet: "))
bedrooms = int(input("Enter the number of bedrooms: "))
age = int(input("Enter the age of the house: "))
location = input("Enter the location (e.g., Peelamedu, Gandhipuram): ").strip()

input_data = pd.DataFrame([[0]*len(model_columns)], columns=model_columns)

input_data.at[0, 'area'] = area
input_data.at[0, 'bedrooms'] = bedrooms
input_data.at[0, 'age'] = age

if location in model_columns:
    input_data.at[0, location] = 1

predicted_price = model.predict(input_data)

print(f"\nPredicted House Price: ₹{predicted_price[0]:,.2f}")