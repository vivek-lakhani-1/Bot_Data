from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
import os
from decouple import config
import sys
from flask_cors import CORS, cross_origin



os.environ['GOOGLE_API_KEY'] = config("GOOGLE_API_KEY")
embeddings=GooglePalmEmbeddings()
PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_API_ENV = config("PINECONE_API_ENV")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV 
)
index_name = "amitava"

docsearch = Pinecone.from_existing_index(index_name, embeddings)


llm = GooglePalm(temperature=0.1)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


app = Flask(__name__)
CORS(app)

@app.route('/',methods=['POST'])
@cross_origin(origin='*')
def getting_data():
    data = request.json
    query = data.get('query', 'World')
    
    response_data = {
        'message': qa({'query': query})['result'],
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run()
