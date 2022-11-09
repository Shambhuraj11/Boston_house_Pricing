import numpy as np
from flask import Flask,request,app,jsonify,url_for,render_template
import joblib


app=Flask(__name__)

#Load Model
model=joblib.load('final_model.pkl')

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    output=int(model.predict(np.array(list(data.values())).reshape(1,-1)))
   
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    final_output=int(model.predict(final_input))
    return render_template('home.html',prediction_text="House price prediction in {}".format(final_output))
    



if __name__=="main":
    app.run(debug=True)
    