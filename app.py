from langchain.vectorstores import Chroma
from src.helper import load_embedding, repo_ingestion
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify, render_template
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY

embeddigs = load_embedding()
persist_directory = "db"

# Now we can load the persisted database from disk and use it 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddigs)

llm  = ChatOpenAI()
memory = ConversationSummaryMemory(llm = llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)


@app.route('/',methods= ["GET","POST"])
def index():
    return render_template('index.html')

@app.route('/chatbot',methods = ["GET", "POST"])
def gitRepo():
    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

@app.route("/get", method = ["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input== "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080, debug=True)