from flask import Flask,render_template,request
import numpy as np
import pickle

knn_model=pickle.load(open("knn_model.pkl","rb"))
smoke_encoder=pickle.load(open("smoke_encoder.pkl","rb"))
gender_encoder=pickle.load(open("gender_encoder.pkl","rb"))

app=Flask(__name__)


def prediction(lst):
    return knn_model.predict(lst)[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        age=int(request.form["age"])
        hyper=int(request.form["hyper"])
        gender=gender_encoder.transform([request.form["gender"]])
        heart=int(request.form["heart"])
        smoke=smoke_encoder.transform([request.form["smoke"]])
        bmi=float(request.form["bmi"])
        HbA1c=float(request.form["HbA1c"])
        glucose=int(request.form["glucose"])

        pred=prediction(np.asarray([[gender,age,hyper,heart,smoke,bmi,HbA1c,glucose]],dtype=object))

        print(f"{age}, {hyper}, {gender}, {heart}, {smoke}, {bmi}, {HbA1c}, {glucose}")

    if pred==0:
        msg="Congratulation!! You don't have diabetes"

    else:
        msg="Sorry!! You have diabetes. Be careful."

    return render_template("index.html",prediction=pred,st=msg)


if __name__=="__main__":
    app.run(debug=True)
