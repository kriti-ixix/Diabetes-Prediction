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
    return render_template('diabetes.html')


#Taking the input from the form
@app.route('/predict', methods=['POST'])
def predict():
    #Getting the input from the form
    bmi = int(request.form['bmi'])
    age = int(request.form['age'])
    glucose = int(request.form['glucose'])
    
    print("Age: ", age)
    print("BMI: ", bmi)
    print("Glucose: ", glucose)

    #Making the predictions
    prediction = loadedModel.predict([[glucose, bmi, age]])[0]
    confidence = loadedModel.predict_proba([[glucose, bmi, age]])

    if prediction == 1:
        sendPrediction = "Diabetic"
    else:
        sendPrediction = "Not Diabetic"

    sendConfidence = str(round((np.amax(confidence[0])*100),2))
    
    print(sendPrediction)
    print(sendConfidence)

    return render_template('diabetes.html', diagnosis_output = sendPrediction, 
                            confidence_output = sendConfidence)


#Main function
if __name__ == '__main__':
    app.run(debug=True)
