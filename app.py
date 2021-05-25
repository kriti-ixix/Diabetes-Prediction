#Importing the libaries 
from flask import Flask, render_template, request, jsonify 
import requests
import pickle
import numpy as np 


#Setting up the API 
app = Flask(__name__)
loadedModel = pickle.load(open('diabetes.sav', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


#Taking the input from the form
@app.route('/predict', methods=['POST'])
def predict():
    #Getting the input
    if request.method=='POST':
        bmi = int(request.form['bmi'])
        age = int(request.form['age'])
        glucose = int(request.form['glucose'])

        #Making predictions
        prediction = loadedModel.predict([[glucose, bmi, age]])  
        confidence = loadedModel.predict_proba([[glucose, bmi, age]]) 

        prediction = int(prediction[0])
        sendConfidence = "Confidence: " + str(round(np.amax(confidence[0])*100, 2))

        if (prediction == 1):
            sendPrediction = "Diagnosis: Diabetic"

        elif (prediction == 0):
            sendPrediction = "Diagnosis: Not diabetic"

        print(sendPrediction)
        print(sendConfidence)

        #Returning the predictions
        return render_template('index.html', diagnosis_text=sendPrediction, confidence_text=sendConfidence)

    else:
        return render_template('index.html')


#Main function
if __name__ == '__main__':
    app.run(debug=True)
