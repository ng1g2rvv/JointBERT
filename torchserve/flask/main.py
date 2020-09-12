from flask import Flask, render_template, request, jsonify
import requests, os
from settings import *


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # curl -X POST http://127.0.0.1:8080/predictions/JointBERT -T sample_pred_in.txt
    if request.method == "POST":
        print(request.form)
        r = requests.post(url="http://127.0.0.1:8080/predictions/JointBERT", data=request.form["text"])
        print(r.text)
        return render_template('index.html', prediction=r.text) 
    else:
        return render_template('index.html', prediction="") 

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
