from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-random-forest-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('Predict.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form.get('slope'))
        ca = int(request.form['ca'])
        thal = int(request.form.get('thal'))

        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
        my_prediction = model.predict(data)

        result_text = "You may have heart disease." if my_prediction[0] == 1 else "You are unlikely to have heart disease."
        
        return render_template('Predict.html', prediction_text=result_text)

    return render_template('Predict.html')

if __name__ == '__main__':
	app.run(debug=True)