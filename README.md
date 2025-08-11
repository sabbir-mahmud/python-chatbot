# 🧠 ChatBot Trainer & Inference

A **simple yet powerful** chatbot pipeline using **NLTK**, **TensorFlow**, and **Bag-of-Words (BoW)** for intent classification.  
It includes a **training script** to prepare data & train a model, and a **chat script** to talk with your trained bot.

---

## 🚀 Features

- **Offline NLTK support** – downloads data to a local folder.  
- **Text preprocessing** – tokenization, lemmatization, and character cleaning.  
- **Bag-of-Words** vectorization.  
- **Neural network classifier** with TensorFlow.  
- **Early stopping** to avoid overfitting.  
- **Persistent storage** of model and vocabulary using `pickle`.  
- **Interactive chat mode** once trained.

---

## 📂 Project Structure

```
├── intents.json         # Training data with patterns & responses
├── model.keras          # Trained TensorFlow model
├── words.pkl            # Vocabulary
├── classes.pkl          # Intent labels
├── train.py             # Training script
├── chat.py              # Chat interface
└── offline/
    └── nltk_data/       # Local NLTK resources
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Run the training script:

```bash
python train.py
```

This will:

1. Load and preprocess data from `intents.json`.  
2. Create training datasets.  
3. Train a **3-layer neural network** with dropout.  
4. Save `model.keras`, `words.pkl`, and `classes.pkl`.

---

## 💬 Talking to the Bot

Once trained, run:

```bash
python chat.py
```

Example:

```
Start chatting with the bot (type 'quit' to stop)!
You: hi
Bot: Hello! How can I help you today?
```

---

## 📜 Intents File Format

The `intents.json` file should look like:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey there"],
      "responses": ["Hello!", "Hi there!", "Hey! How can I help?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you", "Goodbye"],
      "responses": ["Goodbye!", "See you later!", "Bye! Take care."]
    }
  ]
}
```
