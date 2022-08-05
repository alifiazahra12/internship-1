from flask import render_template, request
from app import app
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd


@app.route("/", methods = ["GET", "POST"])
def index():
    msg = None
    if(request.method == "POST"):
        if request.method == 'POST':
            text = request.form['text']
            print(text)
            pickle_in = open("multinomial_naivebayes", "rb")
            model = pickle.load(pickle_in)
            vectorize = pickle.load(open("vectorize", "rb"))
            print(vectorize)
            testing_news = {"text":[text]}
            new_def_test = pd.DataFrame(testing_news)
            new_x_test = new_def_test["text"]
            print(new_x_test)
            new_xv_test = vectorize.transform([text])
            print(new_xv_test)
            preds = model.predict(new_xv_test)
            if preds[0] == 0:
                result = "Fake News"
            elif preds[0] == 1:
                result = "Not A Fake News"
            print(preds)
            return render_template("hasil.html", msg = result)
        else:
            msg = "Username is not available"

    return render_template("index.html", msg = msg)
