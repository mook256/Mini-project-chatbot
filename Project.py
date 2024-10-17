import requests
import json
import logging
import psutil
from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ตั้งค่าระบบ log
logging.basicConfig(level=logging.INFO)

# ตรวจสอบการใช้งานหน่วยความจำ
def check_memory():
    memory_info = psutil.virtual_memory()
    logging.info(f"Memory usage: {memory_info.percent}%")

# ตรวจสอบการใช้งานหน่วยความจำก่อนเริ่มทำงาน
check_memory()

# Initialize the SentenceTransformer model
logging.info("Loading SentenceTransformer model...")
try:
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise e

# Neo4j database connection details
URI = "neo4j://localhost"
AUTH = ("neo4j", "Mook2024")

# Ollama API endpoint (adjust URL if necessary)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}

# Function to send a request to the Ollama API
def send_to_ollama(prompt):
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",  # Updated model name
        "prompt": prompt + "ตอบแบบผู้เชี่ยวชาญสกินแคร์  การดูแลลผิวหน้า ตอบไม่เกิน 20 คำ",
        "stream": False
    }
    logging.info("Sending request to Ollama API...")
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = json.loads(response.text)
        return f"{data['response']}"
    except Exception as e:
        logging.error(f"Error with Ollama API: {e}")
        return f"ไม่สามารถรับข้อมูลจาก Ollama: {response.status_code}, {response.text}"

# Function to run Neo4j queries
def run_query(query, parameters=None):
    logging.info("Running Neo4j query...")
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            with driver.session() as session:
                result = session.run(query, parameters)
                return [record for record in result]
    except Exception as e:
        logging.error(f"Error with Neo4j: {e}")
        raise e

# Load greeting and question corpus from Neo4j
def load_corpora():
    greeting_query = '''
    MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
    '''
    question_query = '''
    MATCH (n:Question) RETURN n.name as name, n.msg_reply as reply;
    '''
    greetings = run_query(greeting_query)
    questions = run_query(question_query)
    
    greeting_corpus = [record['name'] for record in greetings]
    question_corpus = [record['name'] for record in questions]
    
    greeting_dict = {record['name']: record['reply'] for record in greetings}
    question_dict = {record['name']: record['reply'] for record in questions}
    
    return greeting_corpus, question_corpus, greeting_dict, question_dict

# โหลดข้อมูลจาก Neo4j
try:
    greeting_corpus, question_corpus, greeting_dict, question_dict = load_corpora()
    logging.info("Corpora loaded successfully")
except Exception as e:
    logging.error(f"Error loading corpora: {e}")
    raise e

# Function to compute cosine similarity between input sentence and corpus
def compute_response(sentence):
    logging.info(f"Received sentence: {sentence}")
    
    # ตรวจสอบการใช้งานหน่วยความจำ
    check_memory()

    try:
        # Encode the question corpus and the input sentence
        question_vec = model.encode(question_corpus, convert_to_tensor=True, normalize_embeddings=True)
        ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
        
        # Compute cosine similarity
        question_scores = util.cos_sim(question_vec, ask_vec).cpu().numpy()
        
        # Get the index of the most similar question
        max_question_index = np.argmax(question_scores)
        max_question_score = question_scores[max_question_index]
        
        # Check if the similarity score is greater than a threshold (0.5)
        if max_question_score > 0.5:
            Match_question = question_corpus[max_question_index]
            return f"คำตอบจาก Neo4J เกี่ยวกับเรื่องสกินแคร์ใบหน้า\nคำถาม: {sentence}\nคำตอบ: {question_dict.get(Match_question, 'ไม่พบคำตอบในระบบ')}"
        else:
            # Handle greeting
            greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
            greeting_scores = util.cos_sim(greeting_vec, ask_vec).cpu().numpy()
            
            max_greeting_index = np.argmax(greeting_scores)
            max_greeting_score = greeting_scores[max_greeting_index]
            
            if max_greeting_score > 0.5:
                Match_greeting = greeting_corpus[max_greeting_index]
                return f"คำตอบจาก Neo4J คำถามทั่วไป\nคำถาม: {sentence}\nคำตอบ: {greeting_dict.get(Match_greeting, 'ไม่พบคำตอบในระบบ')}"
            else:
                # Send to Ollama API if no match found
                return f"คำตอบจาก Ollama คำถามนอกขอบเขต\nคำถาม: {sentence}\nคำตอบ: " + send_to_ollama(sentence)
    except Exception as e:
        logging.error(f"Error in compute_response: {e}")
        return "เกิดข้อผิดพลาดในการประมวลผล"

# Flask app to handle Line bot messages
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    logging.info("Received request from Line bot")
    body = request.get_data(as_text=True)
    
    try:
        json_data = json.loads(body)
        access_token = 'T2wYXo6uT6QQ/8A/zyD4mYi857RA8hQv/CkVT1q24h0kA6xAvD0tewp6Z56DFir1+XdImm7Av/ZiL4gFAOLA7rY5DG/yHhsyTnwlZAyWFxINNWxFsW7Q5zpTcLj/UHr6N72zk5B5nkQiME/NkR7F5gdB04t89/1O/w1cDnyilFU='
        secret = '2da1d9cc29b8b9a0fc27224ac051a6b5'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        
        # Get the message and reply token from the JSON data
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        
        # Compute the response based on the message
        response_msg = compute_response(msg)
        
        # Reply to the user via Line
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
        logging.info(f"Message: {msg}, ReplyToken: {tk}")
    except Exception as e:
        logging.error(f"Error in linebot: {e}")
        logging.error(f"Request body: {body}")
    return 'OK'

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(port=5000)
