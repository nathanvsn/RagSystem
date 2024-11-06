from flask import Flask, render_template, request, jsonify
from main import ChatSession  # Supondo que ChatSession está em chat_session.py

app = Flask(__name__)
chat = ChatSession()  # Inicializa a sessão de chat

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message')
    if user_message:
        chat.add_user_message(user_message)  # Adiciona mensagem do usuário
        response = chat.get_response()  # Gera resposta da IA
        return jsonify({'user_message': user_message, 'ai_response': response})
    return jsonify({'error': 'Mensagem vazia'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8888, host='192.168.100.34')
