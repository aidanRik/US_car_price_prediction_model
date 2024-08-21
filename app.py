from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('price_prediction_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    print()
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get Original Price
        Price = float(request.form['price'])

        # Get Year
        Year = int(request.form['year'])
        Year = 2024 - Year  # Car age calculation based on 2024

        # Get Mileage
        Mileage = int(request.form['mileage'])
        Mileage_log = np.log(Mileage)  # Log transform mileage for prediction

        # Get Condition
        Condition = request.form['condition']
        condition_mapping = {'new': 1, 'poor': 0}  # Map 'New' -> 1, 'Poor' -> 0
        Condition_num = condition_mapping.get(Condition, -1)

        # Get Color
        Color = request.form['color']
        color_mapping = {'black': 0, 'white': 1, 'red': 2, 'blue': 3, 'silver': 4, 'gray': 5, 'green': 6}
        Color_num = color_mapping.get(Color, -1)

        # Predict the price using the model
        prediction = model.predict([[Price, Year, Color_num, Mileage_log, Condition_num]])

        # Format the output
        output = round(prediction[0], 2)

        # Return the result based on the prediction
        if output < 0:
            return render_template('index.html', prediction_texts="Sorry, you cannot sell this car.")
        else:
            return render_template('index.html', prediction_text="You can sell the car at ${}".format(output))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
