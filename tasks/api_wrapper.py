from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OPENAI_API_KEY = "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"
OPENAI_API_BASE_URL = "https://api.openai.com/v1/chat/completions"

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    model = data.get('model', '')

    if 'gpt' in model:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        response = requests.post(OPENAI_API_BASE_URL, headers=headers, json=data)
    else:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer EMPTY'
        }
        response = requests.post('http://localhost:8000', headers=headers, json=data)

    return jsonify(response.json()), response.status_code

if __name__ == '__main__':
    app.run(port=8002)
