from flask import Flask,render_template,request
import pickle


app=Flask(__name__)


model=pickle.load(open("flood_model.pkl","rb"))


@app.route("/")
def home():

    return render_template("index.html")



@app.route("/predict",methods=["POST"])

def predict():

    rainfall=float(request.form["rainfall"])

    temp=float(request.form["temperature"])

    humidity=float(request.form["humidity"])

    water=float(request.form["water"])


    result=model.predict(
        [[rainfall,temp,humidity,water]]
    )


    if result[0]==1:

        output="High Flood Risk"

    else:

        output="Low Flood Risk"


    return render_template(
        "index.html",
        prediction=output
    )


if __name__=="__main__":

    app.run(debug=True)