  
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    # return "<h1 style='color:blue'>Baseline Main Page</h1>"
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/inference')#,methods=['POST'])
def inference():
    return render_template("inference.html")#, prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict')
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[7],2)

    return render_template('index.html', prediction_text='our condition prediction is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)