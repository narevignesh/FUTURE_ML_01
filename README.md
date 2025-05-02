# Customer Support Chatbot

A modern customer support chatbot built with Transformers, TensorFlow, NLTK, and Streamlit.

## Features

- Natural Language Processing using NLTK
- Deep Learning model using Transformers and TensorFlow
- Modern and responsive UI using Streamlit
- Real-time chat interface
- Text preprocessing and tokenization
- Context-aware responses

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd customer-support-chatbot
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Start chatting with the bot!

## How it Works

The chatbot uses:
- NLTK for text preprocessing and tokenization
- Transformers (DistilBERT) for understanding user queries
- TensorFlow for model inference
- Streamlit for the web interface

## Customization

You can customize the chatbot by:
- Modifying the response templates in `app.py`
- Training the model on your specific dataset
- Adjusting the UI elements in the Streamlit interface

## License

This project is licensed under the MIT License - see the LICENSE file for details. 