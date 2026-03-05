from flask import Flask, render_template, request, redirect
import joblib
import numpy as np
import json

app = Flask(__name__)
model = joblib.load('house_price_model.pkl')

# Load columns.json
with open("columns.json", "r") as f:
    columns = json.load(f)

@app.route('/')
def home():
    return redirect('/login')  # Redirects to login page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "1234":
            return redirect('/predict')
        else:
            return render_template('login.html', error="❌ Invalid username or password")
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])
        location = request.form['location']

        # Create input vector for model
        x = np.zeros(len(columns))
        x[columns.index('area')] = area
        x[columns.index('bedrooms')] = bedrooms
        x[columns.index('age')] = age

        if location in columns:
            x[columns.index(location)] = 1

        prediction = model.predict([x])[0]

        # Show result in new page with image
        return render_template('result.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)