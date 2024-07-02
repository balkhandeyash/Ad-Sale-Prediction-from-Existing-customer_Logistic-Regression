from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

app = Flask(__name__)

# Load Dataset
dataset = pd.read_csv('DigitalAd_dataset.csv')

# Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting Dataset into Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    salary = float(request.form['salary'])

    # Create a DataFrame for prediction
    new_customer_data = pd.DataFrame({'Age': [age], 'Salary': [salary]})
    new_customer_data_scaled = sc.transform(new_customer_data)

    result = model.predict(new_customer_data_scaled)
    if result == 1:
        prediction = "Customer will Buy"
    else:
        prediction = "Customer won't Buy"
    
    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)
