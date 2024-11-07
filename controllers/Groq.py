# qa_pipeline/groq_integration.py
import os
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

import logging

logging.basicConfig(level=logging.INFO)

# Configuração de API e banco de dados
os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")

class GroqIntegration:
    def __init__(self):
        self.__chat = ChatGroq(model=config("GROQ_MODEL"))
        self.__retriever = self.__build_retriever()
    
    def __build_retriever(self):
        directory = 'data/chroma'
        embedding = HuggingFaceEmbeddings(model_name=config("HUGGINGFACE_MODEL"))
        
        return Chroma(
            persist_directory=directory,
            embedding_function=embedding,
        )
    
    def __retrieve_documents(self, user_message):
        # Realiza a consulta usando similarity_search_with_score para obter documentos, metadados e distâncias
        results = self.__retriever.similarity_search_with_score(
            query=user_message,
            k=10  # Número de resultados que deseja retornar
        )
        
        # Processa os resultados, convertendo-os em objetos Document
        docs = []
        for result in results:
            document, score = result  # 'score' representa a distância ou relevância
            doc_data = Document(
                page_content=document.page_content,
                metadata=document.metadata
            )
            docs.append(doc_data)
        
        return docs

            
    def __build_messages(self, history_messages, user_message):
        messages = []
        for message in history_messages:
            message_class = HumanMessage if message['author'] == 'user' else AIMessage
            messages.append(message_class(content=message.get('body', '')))
        messages.append(HumanMessage(content=user_message))
        return messages
    

    def invoke(self, history_messages, user_message):
        SYSTEM_PROMPT = '''Você é um advogado focado em mercado de capitais. Seu objetivo é auxiliar com respostas diretas e informativas.
        Seja claro e objetivo, usando linguagem formal e técnica de forma humanizada como se estivesse conversando com um cliente.
        Responda sempre em português e considere o contexto da empresa, quando aplicável.
        
        Reponda de acordo com o contexto, caso o contexto não tenha a informação que responda a pergunta, fala que não sabe ou que não possui a informação.
        Não Inventa a informação.
        
        Não fale sobre o contexto, o usuario não sabe oq acontece no backend.
        
        <context>
        {context}
        </context>
        '''
        
        docs = self.__retrieve_documents(user_message)

        question_answering_prompt = ChatPromptTemplate.from_messages([
            'system',
            SYSTEM_PROMPT,
            MessagesPlaceholder(variable_name='messages'),
        ])
        
        document_chain = create_stuff_documents_chain(
            llm=self.__chat,
            prompt=question_answering_prompt,
            document_variable_name='context'
        )
        
        response = document_chain.invoke({
            'context': docs,
            'messages': self.__build_messages(history_messages, user_message),
        })
                
        return response

