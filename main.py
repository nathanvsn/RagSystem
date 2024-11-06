# qa_pipeline/chat_session.py
from controllers.Groq import GroqIntegration
import logging

# Guardar em logs/ 
# logging.basicConfig(filename='logs/chat_session.log', level=logging.DEBUG)

class ChatSession:
    def __init__(self):
        self.integration = GroqIntegration()  # Inicializa o GroqIntegration
        self.history = []

    def add_user_message(self, message):
        """Adiciona uma mensagem do usuário ao histórico."""
        self.history.append({'author': 'user', 'body': message})
    
    def get_response(self):
        """Gera uma resposta usando o modelo e adiciona ao histórico."""
        if not self.history:
            raise ValueError("Histórico vazio! Adicione uma mensagem do usuário primeiro.")
        
        # Última mensagem do usuário
        user_message = self.history[-1]['body']
        
        # Gera a resposta do modelo
        response = self.integration.invoke(self.history, user_message)
        
        # Adiciona a resposta do modelo ao histórico
        self.history.append({'author': 'AI', 'body': response})
        return response
    
    def show_conversation(self):
        """Exibe o histórico da conversa."""
        for message in self.history:
            author = "Usuário" if message['author'] == 'user' else "IA"
            print(f"{author}: {message['body']}")

# Exemplo de uso
if __name__ == '__main__':
    chat = ChatSession()

    while True:
        user_input = input("Usuário: ")
        chat.add_user_message(user_input)
        
        response = chat.get_response()
        print(f"IA: {response}")
        
        if user_input.lower() == "sair":
            break