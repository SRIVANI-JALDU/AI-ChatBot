from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI, APIConnectionError, RateLimitError, APIError, AuthenticationError
import os
import random
from dotenv import load_dotenv
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

load_dotenv()

# Initialize OpenAI client with error handling
client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI initialized successfully")
    else:
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
except Exception as e:
    logger.error(f"❌ OpenAI initialization failed: {e}")

# Enhanced local responses
LOCAL_KNOWLEDGE = {
    "greetings": [
        "Hello! I'm your AI assistant. How can I help you today?",
        "Hi there! What would you like to know?",
        "Greetings! I'm ready to answer your questions."
    ],
    "goodbye": [
        "Goodbye! Come back if you have more questions.",
        "See you later! I'm here if you need me."
    ],
    "thanks": [
        "You're welcome! Happy to help.",
        "My pleasure! Let me know if you need anything else."
    ],
    "capabilities": [
        "I can answer questions on many topics including technology, science, and history.",
        "My knowledge covers various subjects. Ask me anything!"
    ],
    "error": [
        "I'm having some technical difficulties. Please try again later.",
        "There seems to be a connection issue. Maybe ask me something else?"
    ],
    "default": [
        "I'm not sure about that. Could you ask me something else?",
        "Interesting question! I'm currently limited in my responses.",
        "Let me think about that... (Note: I'm using local responses right now)"
    ]
}

def get_local_response(user_input):
    """Get an appropriate local response"""
    user_input = user_input.lower().strip()
    
    if not user_input:
        return "Please type something so I can help you."
    
    # Check for specific types of input
    if any(word in user_input for word in ["hi", "hello", "hey"]):
        return random.choice(LOCAL_KNOWLEDGE["greetings"])
    if any(word in user_input for word in ["bye", "goodbye", "see you"]):
        return random.choice(LOCAL_KNOWLEDGE["goodbye"])
    if any(word in user_input for word in ["thank", "thanks"]):
        return random.choice(LOCAL_KNOWLEDGE["thanks"])
    if any(word in user_input for word in ["what can you do", "capabilities"]):
        return random.choice(LOCAL_KNOWLEDGE["capabilities"])
    if "?" in user_input:  # If it's a question
        return random.choice(LOCAL_KNOWLEDGE["default"])
    
    return random.choice(LOCAL_KNOWLEDGE["default"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        user_input = data['message'].strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        logger.info(f"Received message: {user_input}")

        # Try OpenAI if available
        if client:
            try:
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                reply = response.choices[0].message.content
                processing_time = round(time.time() - start_time, 2)
                
                logger.info(f"OpenAI response time: {processing_time}s")
                
                return jsonify({
                    "reply": reply,
                    "source": "openai",
                    "status": "success",
                    "processing_time": processing_time
                })
                
            except AuthenticationError:
                logger.error("OpenAI authentication failed - check your API key")
                return jsonify({
                    "reply": "API authentication failed. Using local response.",
                    "source": "local",
                    "status": "auth_error"
                })
                
            except RateLimitError:
                logger.warning("OpenAI rate limit exceeded")
                return jsonify({
                    "reply": get_local_response(user_input),
                    "source": "local",
                    "status": "rate_limit"
                })
                
            except APIConnectionError:
                logger.warning("OpenAI connection error")
                return jsonify({
                    "reply": get_local_response(user_input),
                    "source": "local",
                    "status": "connection_error"
                })
                
            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                return jsonify({
                    "reply": get_local_response(user_input),
                    "source": "local",
                    "status": "api_error"
                })
                
            except Exception as e:
                logger.error(f"Unexpected OpenAI error: {e}")
        
        # Fallback to local responses
        logger.info("Using local response")
        return jsonify({
            "reply": get_local_response(user_input),
            "source": "local",
            "status": "fallback"
        })
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({
            "reply": "Sorry, I encountered an unexpected error. Please try again.",
            "source": "error",
            "status": "server_error"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)