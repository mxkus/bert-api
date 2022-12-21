from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"],
)


@app.route("/bert/", methods=["GET", "POST"])
@app.before_request
@limiter.limit("10 per minute")
def answer():

    from transformers import AutoTokenizer, MobileBertForQuestionAnswering, pipeline
    import requests

    modelname = "csarron/mobilebert-uncased-squad-v2"

    model = MobileBertForQuestionAnswering.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": "Angela_Merkel",
            "prop": "extracts",
            "explaintext": True,
        },
    )

    json_content = response.json()["query"]["pages"]

    context = ""
    for key in json_content.keys():
        context += json_content[key]["extract"]

    response_bert = nlp({"question": request.json["question"], "context": context})
    answer = response_bert.get("answer", "No answer available.")
    confidence = response_bert.get("score", "No confidence available")
    response = app.response_class(response=f"{answer} (Confidence: {confidence:.4f})", status=200, mimetype="application/json")

    return response


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=1025)
