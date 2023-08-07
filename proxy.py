import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def proxy():
    # Determine where to send the request
    model = request.json.get('model')
    if model and ('llama' in model or "Llama" in model):
        # Send the request to the local server
        url = 'http://localhost:8000/v1/chat/completions'
        headers = {
            'Authorization': 'Bearer EMPTY', 
            'Content-Type': 'application/json'
        }
    else:
        # Send the request to the OpenAI server
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}', 
            'Content-Type': 'application/json'
        }

    response = forward_request(url, headers)
    
    # Return the response
    return jsonify(response.json()), response.status_code

def forward_request(url, headers):
    method = request.method.lower()

    if method == 'post':
        return requests.post(url, headers=headers, json=request.json)
    else:
        return None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
