# 🧠 Sentiment Aware Chatbot

Sentiment Aware Chatbot is a full-stack AI-powered conversational system designed to create empathetic and context-aware interactions. The chatbot combines Natural Language Processing (NLP), Human-Computer Interaction (HCI), and Machine Learning techniques to understand user emotions and generate meaningful responses.

The system analyzes sentiment using a fine-tuned BERT model and responds through predefined emotion-aligned responses or BlenderBot-generated conversational outputs.

---

## 🌟 Features

### Core AI Features

💬 Real-time sentiment detection using Fine-Tuned BERT  
🧠 Emotion-aware response generation  
🤖 Dynamic response generation using BlenderBot  
📊 Feedback logging system for user interaction tracking  
📁 Custom CSV-based response matching  
⚡ Fast Flask API backend  
🎨 Responsive React-based user interface  

---

## User Experience Features

😊 Greeting detection and personalized interactions  
📈 Continuous response improvement through feedback logging  
🔍 Context-aware conversational flow  
📱 Responsive design across desktop and mobile devices  

---

## 🚀 Tech Stack

### Frontend

- React.js
- HTML
- CSS
- JavaScript

### Backend

- Flask
- REST APIs
- Python

### NLP / AI

- Fine-Tuned BERT
- Hugging Face Transformers
- BlenderBot
- Sentiment Analysis

### Data Processing

- Pandas
- PyTorch

---

## 📦 Installation

### Prerequisites

- Python >= 3.9
- Node.js >= 18
- npm
- Git

---

## Quick Start

Clone repository:

```bash
git clone https://github.com/akankshabolla/Sentiment_aware_chatbot.git

cd Sentiment_aware_chatbot
```

---

## Backend Setup

Move to backend:

```bash
cd server
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run backend:

```bash
python app.py
```

Backend starts:

```text
http://localhost:5000
```

---

## Frontend Setup

Open another terminal:

```bash
cd client
```

Install packages:

```bash
npm install
```

Run frontend:

```bash
npm start
```

Frontend starts:

```text
http://localhost:3000
```

---

## 📁 Project Structure

```text
Sentiment_aware_chatbot/
│
├── client/                    # React frontend
│
│   ├── public/
│   ├── src/
│   └── package.json
│
├── server/                    # Flask backend
│   ├── app.py
│   ├── custom_responses.csv
│   ├── requirements.txt
│   └── fine_tuned_sentiment_model/
│
├── training/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── evaluation_figures/
│   └── training_report.txt
│
├── data/                      # Dataset files
│
├── .gitignore
└── README.md
```

---

## 🔌 API Endpoints

### Chat API

| Method | Endpoint | Description |
|----------|-----------|-------------|
| POST | `/chat` | Generate chatbot response |

Request:

```json
{
 "message":"I feel sad today"
}
```

Response:

```json
{
 "response":"I'm sorry you're feeling that way.",
 "sentiment":"negative"
}
```

---

### Feedback API

| Method | Endpoint | Description |
|----------|-----------|-------------|
| POST | `/feedback` | Store user feedback |

Request:

```json
{
"user_input":"I feel stressed",
"bot_response":"Take some time to relax.",
"sentiment":"negative",
"feedback":"helpful"
}
```

---

## 🧠 System Architecture

User Input  
↓  
React Frontend  
↓  
Flask API  
↓  
Fine-Tuned BERT  
↓  
Emotion Detection  
↓  
Custom Response Matching  
↓  
BlenderBot Fallback  
↓  
Final Response Returned  

---

## 📊 Model Information

The chatbot uses a Fine-Tuned BERT model with:

Sentiment Labels:

- Positive
- Neutral
- Negative

Response Flow:

1. Detect sentiment
2. Search custom response CSV
3. Generate BlenderBot response if no match found
4. Return response to frontend

---

## 📸 Screenshots

Add screenshots after uploading:

```text
screenshots/chat_ui.png
```

Example:

```markdown
![Chat Interface](screenshots/chat_ui.png)
```

---

## 🔮 Future Improvements

🚧 Multi-emotion classification  
🚧 Long-term conversation memory  
🚧 Voice interaction support  
🚧 User profile personalization  
🚧 LLM integration (Gemini/OpenAI)  
🚧 Deployment using Render + Vercel  
🚧 Multi-language support  

---

## 👩‍💻 Author

Akanksha Bolla

Master's in Computer Science  
Texas A&M University–Corpus Christi



---

## ⭐ Support

If you found this project useful, consider giving it a star.
