#!/usr/bin/env python3
"""
LLM Question-and-Answering Web GUI Application
Author: SORETIRE OLUWANIFEMI
Matric Number: 23ch034242

Flask-based web application for Q&A using LLM API.
"""

import os
import string
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# Text Preprocessing Functions
def preprocess_text(text):
    """
    Apply basic preprocessing to the input text:
    - Lowercasing
    - Tokenization
    - Punctuation removal
    """
    # Lowercasing
    text_lower = text.lower()

    # Remove punctuation
    text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))

    # Tokenization (split into words)
    tokens = text_no_punct.split()

    # Join tokens back for display
    processed_text = ' '.join(tokens)

    return processed_text, tokens

def construct_prompt(question):
    """
    Construct a prompt for the LLM API
    """
    prompt = f"""You are a helpful assistant. Please answer the following question clearly and concisely.

Question: {question}

Answer:"""
    return prompt

def get_llm_response(question):
    """
    Send the question to the Groq LLM API and get the response
    """
    # Initialize the Groq client
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return "Error: GROQ_API_KEY not configured. Please set the environment variable."

    client = Groq(api_key=api_key)

    # Construct the prompt
    prompt = construct_prompt(question)

    # Send request to the API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024
    )

    # Extract and return the response
    return chat_completion.choices[0].message.content

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle the question submission"""
    try:
        # Get the question from the request
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({
                'success': False,
                'error': 'Please enter a valid question.'
            })

        # Preprocess the question
        processed_question, tokens = preprocess_text(question)

        # Get response from LLM
        answer = get_llm_response(question)

        return jsonify({
            'success': True,
            'original_question': question,
            'processed_question': processed_question,
            'tokens': tokens,
            'answer': answer
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
