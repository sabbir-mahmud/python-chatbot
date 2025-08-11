import json  # Module for reading/writing JSON data
import pickle  # Module for saving/loading Python objects in binary format
import random  # Module for random number generation
from pathlib import Path  # Object-oriented file system paths
from typing import Dict, List, Tuple  # Type hints for better code clarity

import nltk  # Natural Language Toolkit for text processing
import numpy as np  # NumPy for numerical computations
import tensorflow as tf  # TensorFlow for building and training neural networks
from nltk.stem import WordNetLemmatizer  # Lemmatizer to reduce words to base form
from nltk.tokenize import word_tokenize  # Function to split text into individual words

# Set up a custom local directory for storing NLTK data
NLTK_DATA_DIR = Path("./offline/nltk_data")

# Create the directory if it doesn't already exist
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Add the custom directory to NLTK's search paths
nltk.data.path.insert(0, str(NLTK_DATA_DIR))


# Function to ensure required NLTK datasets are available locally
def download_nltk_resources():
    resources = ["punkt", "wordnet", "omw-1.4"]  # Required datasets
    for resource in resources:
        try:
            # Check if the resource exists locally
            if resource == "punkt":
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            # Download the missing resource into the local directory
            nltk.download(resource, download_dir=str(NLTK_DATA_DIR))


# Download missing NLTK resources
download_nltk_resources()

# Set a random seed for reproducibility across Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Function to load intents data from a JSON file
def load_intents(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")  # Read the file content
    return json.loads(text)  # Parse and return as a Python dictionary


# Function to preprocess intents data and extract vocabulary, classes, and training documents
def preprocess_intents(
    intents: Dict, ignore_letters: set = None
) -> Tuple[List[str], List[str], List[Tuple[List[str], str]]]:
    if ignore_letters is None:
        ignore_letters = {"?", "!", ".", ","}  # Characters to ignore
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    words = []  # Store vocabulary words
    classes = []  # Store intent class labels
    documents = []  # Store training data as (token_list, intent_tag)

    # Iterate over each intent in the dataset
    for intent in intents.get("intents", []):
        tag = intent.get("tag")  # Get the intent tag
        if not tag:
            continue
        if tag not in classes:
            classes.append(tag)  # Add tag to class list if new

        # Process each training pattern in the intent
        for pattern in intent.get("patterns", []):
            tokens = word_tokenize(pattern)  # Tokenize the pattern
            tokens = [
                lemmatizer.lemmatize(tok.lower())  # Lemmatize and lowercase each token
                for tok in tokens
                if tok not in ignore_letters
            ]
            words.extend(tokens)  # Add tokens to vocabulary list
            documents.append((tokens, tag))  # Store (tokens, tag) pair

    # Remove duplicates and sort the lists
    words = sorted(set(words))
    classes = sorted(set(classes))
    return words, classes, documents


# Function to build training data arrays from vocabulary, classes, and documents
def build_training_data(
    words: List[str], classes: List[str], documents: List[Tuple[List[str], str]]
):
    training = []  # Store all training samples
    output_empty = [0] * len(classes)  # Template for output one-hot vector

    # Create bag-of-words and output label for each document
    for pattern_tokens, tag in documents:
        bag = [1 if w in pattern_tokens else 0 for w in words]  # Bag-of-words vector
        output_row = output_empty.copy()
        output_row[classes.index(tag)] = 1  # Mark correct intent class
        training.append(bag + output_row)  # Combine input and output vectors

    # Shuffle the training data for randomness
    random.shuffle(training)
    # Convert to NumPy array
    training = np.array(training, dtype=np.float32)
    # Split into input (X) and output (Y) arrays
    trainX = training[:, : len(words)]
    trainY = training[:, len(words) :]
    return trainX, trainY


# Function to build a neural network model for intent classification
def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, input_shape=(input_dim,), activation="relu"),
            tf.keras.layers.Dropout(0.5),  # Dropout for regularization
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(output_dim, activation="softmax"),  # Output layer
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Function to train the model with training data
def train_model(
    model: tf.keras.Model,
    trainX: np.ndarray,
    trainY: np.ndarray,
    epochs: int = 200,
    batch_size: int = 5,
):
    # Stop training early if no improvement in loss for 'patience' epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=8, restore_best_weights=True
    )
    # Train the model and store history
    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# Main execution block
if __name__ == "__main__":
    intents_path = Path("intents.json")  # Path to intents JSON file
    intents = load_intents(intents_path)  # Load intents data
    words, classes, documents = preprocess_intents(intents)  # Preprocess data
    # Save vocabulary and classes to files for later use
    with open("words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open("classes.pkl", "wb") as f:
        pickle.dump(classes, f)
    # Prepare training input and output arrays
    trainX, trainY = build_training_data(words, classes, documents)
    # Build the neural network model
    model = build_model(input_dim=trainX.shape[1], output_dim=trainY.shape[1])
    # Train the model
    history = train_model(model, trainX, trainY, epochs=200, batch_size=5)
    # Save the trained model to file
    model.save("model.keras")
    print("Done")
