from flask import Flask, render_template, request, jsonify, send_from_directory
from main import ChatSession  # Supondo que ChatSession está em chat_session.py
from Rag.rag import RagSystem
import os

app = Flask(__name__)
chat = ChatSession()
rag = RagSystem()

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

@app.route('/send_file_to_rag', methods=['POST'])
def send_file_to_rag():
    if 'files' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['files']

    # Verificar se o arquivo é um PDF
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400

    pdf_path = os.path.join('data/inputpdfs', file.filename)  # Caminho onde o PDF será salvo

    try:
        file.save(pdf_path)  # Salvar o arquivo enviado

        # Extrair metadados da requisição
        document_metadata = request.form.to_dict()

        # Exemplo de validação dos metadados obrigatórios
        required_fields = ['document_type', 'document_number']
        for field in required_fields:
            if field not in document_metadata:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Chamada ao método run da classe RagSystem
        rag.run(
            pdf_path=pdf_path,
            group_size=int(document_metadata.get('group_size', 5)),
            threshold=float(document_metadata.get('threshold', 0.7)),
            chunk_size=int(document_metadata.get('chunk_size', 1000)),
            chunk_overlap=int(document_metadata.get('chunk_overlap', 250)),
            directory='data/chroma',
            document_metadata=document_metadata
        )

        return jsonify({'message': 'File processed successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_vectorized_files', methods=['GET'])
def get_vectorized_files():
    def list_files(directory):
        import os
        # Retorna uma lista de arquivos .pdf no diretório
        return [f for f in os.listdir(directory) if f.endswith('.pdf')]

    directory = 'data/inputpdfs'  # O diretório onde os PDFs estão salvos
    arquivos = list_files(directory)  # Obtém a lista de arquivos

    # Construir uma lista de dicionários para os arquivos
    vectorized_files = []
    for filename in arquivos:
        document_type = filename.split('.')[-1]
        document_number = filename.split('.')[0] 

        vectorized_files.append({
            'filename': filename,
            'document_type': document_type,
            'document_number': document_number,
        })

    return render_template('_arquivos_disponiveis.html', arquivos=vectorized_files)

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory('data/inputpdfs', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=8888, host='192.168.100.34')
