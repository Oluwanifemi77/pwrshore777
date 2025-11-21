#!/usr/bin/env python3
"""
LLM Question-and-Answering CLI Application
Author: SORETIRE OLUWANIFEMI
Matric Number: 23ch034242

This CLI application accepts natural-language questions, processes them,
sends them to an LLM API (Groq), and displays the final answer.
"""

import os
import re
import string
from groq import Groq

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
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
        model="llama-3.1-8b-instant",  # Using Llama 3.1 model
        temperature=0.7,
        max_tokens=1024
    )

    # Extract and return the response
    return chat_completion.choices[0].message.content

def main():
    """
    Main function to run the CLI Q&A application
    """
    print("=" * 60)
    print("   NLP Question-and-Answering System")
    print("   Author: SORETIRE OLUWANIFEMI | Matric: 23ch034242")
    print("=" * 60)
    print()

    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set it using: export GROQ_API_KEY='your-api-key'")
        print("Get your free API key at: https://console.groq.com/keys")
        return

    while True:
        print("-" * 60)
        # Get user input
        question = input("\nEnter your question (or 'quit' to exit): ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Q&A System. Goodbye!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        # Preprocess the question
        processed_question, tokens = preprocess_text(question)

        print("\n--- Processing ---")
        print(f"Original Question: {question}")
        print(f"Processed Question: {processed_question}")
        print(f"Tokens: {tokens}")

        try:
            # Get response from LLM
            print("\n--- Fetching Answer from LLM ---")
            answer = get_llm_response(question)

            print("\n--- Answer ---")
            print(answer)
            print()

        except Exception as e:
            print(f"\nError getting response: {str(e)}")
            print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main()
