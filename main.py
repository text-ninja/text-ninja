# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_app]
from flask import Flask
from flask import request
from nlp.utils import compute_similarity
import json

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.route('/create', methods=['POST'])
def clone():
    # with open('ak.txt', 'rb') as content_file:
    #     content = content_file.read().decode(errors='replace')
    # print(content.split("\r\n"))
    # ids = compute_similarity(content.split("\n"), [0])
    # print(ids)
    paraJson = request.get_data()
    data = json.loads(paraJson)

    ids = compute_similarity(data["paras"], data["index"])

    # print(paraJson)
    id_json = {}
    id_json = ids
    val = json.dumps(id_json)
    return val

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
