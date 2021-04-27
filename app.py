import os
import traceback
from flask import Flask
from flask import render_template, json, request, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import datetime
import pickle
import logging

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"],
)


@app.route('/bert/', methods=['GET', 'POST'])
@app.before_request
@limiter.limit("10 per minute")
def answer():

    from transformers import MobileBertForQuestionAnswering, AutoTokenizer

    modelname = 'csarron/mobilebert-uncased-squad-v2'

    model = MobileBertForQuestionAnswering.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    from transformers import pipeline
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    with open("merkel.txt", "r") as file:
        context = file.read()

    response_bert = nlp({
        'question': request.json["question"],
        'context': context
    })
    response = app.response_class(
        response=response_bert.get("answer", "No answer available."),
        status=200,
        mimetype='application/json'
    )

    return response




if __name__ == "__main__":

    app.run(host='0.0.0.0', port=1025)

