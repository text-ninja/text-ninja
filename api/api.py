import flask
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/create', methods=['POST'])
def clone():
    paras = request.args.get('data')
    return "Hello"
app.run()