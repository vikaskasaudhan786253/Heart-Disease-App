from flask import Flask,render_template,request
import pickle
from lightgbm import LGBMClassifier
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    age = int(request.form['age'])
    sex = int(request.form['sex'])
    chest_pain = int(request.form['chest_pain_type'])
    bp = float(request.form['blood_pressure'])
    choles = float(request.form['cholesterol'])
    fbs = int(request.form['fbs'])
    ekg = int(request.form['ekg'])
    max_heart_rate = float(request.form['max_heart_rate'])
    angina = int(request.form['angina'])
    st_depression = float(request.form['st_depression'])
    slope_st = int(request.form['st_slope'])
    vessels = int(request.form['num_major_vessels'])
    thallium = int(request.form['thallium'])

    features = [age, sex, chest_pain, bp, choles, fbs, ekg,
                max_heart_rate, angina, st_depression,
                slope_st, vessels, thallium]

    prediction = model.predict([features])
    prob = model.predict_proba([features])[0][0]*100

    result = [round(prob,2),prediction]

    return render_template('predict.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
