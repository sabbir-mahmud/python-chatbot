import json  # Module for working with JSON data (read/write)
import pickle  # Module for serializing and deserializing Python objects

import numpy as np  # NumPy for working with arrays and numerical operations
import tensorflow as tf  # TensorFlow for loading and using the trained neural network model
from nltk.stem import (
    WordNetLemmatizer,
)  # Lemmatizer for reducing words to their base form
from nltk.tokenize import word_tokenize  # Tokenizer for splitting text into words

# Load the pre-trained model from the file 'model.keras'
model = tf.keras.models.load_model("model.keras")
# Load the vocabulary list from 'words.pkl' (serialized Python object)
words = pickle.load(open("words.pkl", "rb"))
# Load the list of classes (intents) from 'classes.pkl'
classes = pickle.load(open("classes.pkl", "rb"))
# Load the intents dataset from 'intents.json' (contains patterns and responses)
intents_json = json.load(open("intents.json", "r", encoding="utf-8"))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
# Set of characters to ignore during tokenization
ignore_letters = {"?", "!", ".", ","}


def clean_up_sentence(sentence):
    # Tokenize the input sentence into words
    sentence_words = word_tokenize(sentence)
    # Lemmatize each word, convert to lowercase, and ignore unwanted characters
    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
        if word not in ignore_letters
    ]
    return sentence_words  # Return the cleaned and lemmatized list of words


def bow(sentence, words, show_details=False):
    # Convert a sentence into a bag-of-words vector
    sentence_words = clean_up_sentence(sentence)
    # Initialize a vector of 0's with the same length as the vocabulary
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:  # If the word is found in the vocabulary
                bag[i] = 1  # Mark presence in the bag
                if show_details:  # Optionally print debug info
                    print(f"Found in bag: {w}")
    return np.array(bag)  # Return as NumPy array


def predict_class(sentence, model):
    # Predict the intent of a given sentence
    p = bow(sentence, words)  # Convert sentence to bag-of-words
    # Get prediction probabilities from the model
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25  # Minimum probability to consider
    # Filter predictions above the threshold
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        # Append dictionary with intent and probability
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list  # Return list of predicted intents


def get_response(intents_list, intents_json):
    # Get an appropriate response based on predicted intents
    if not intents_list:  # If no intent predicted
        return "Sorry, I didn't understand that."
    tag = intents_list[0]["intent"]  # Get the highest probability intent
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            # Randomly select a response from the matched intent
            return np.random.choice(i["responses"])
    return "Sorry, I didn't understand that."  # Fallback response


def chat():
    # Main chat loop for interacting with the bot
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        inp = input("You: ")  # Get user input
        if inp.lower() == "quit":  # Exit condition
            break
        predicted_intents = predict_class(inp, model)  # Predict intent
        response = get_response(predicted_intents, intents_json)  # Get bot response
        print("Bot:", response)  # Display bot response


if __name__ == "__main__":
    chat()  # Run the chatbot when the script is executed
