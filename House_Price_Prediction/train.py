import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import json

data = pd.read_csv('data.csv')

location_dummies = pd.get_dummies(data['location'], drop_first=True)

x = pd.concat([data[['area', 'bedrooms', 'age']], location_dummies], axis=1)
y = data['price']

model = LinearRegression()
model.fit(x, y)

joblib.dump(model, 'house_price_model.pkl')

model_columns = list(x.columns)
with open("columns.json", "w") as f:
    json.dump(model_columns, f)

print("Model trained and saved as 'house_price_model.pkl'")