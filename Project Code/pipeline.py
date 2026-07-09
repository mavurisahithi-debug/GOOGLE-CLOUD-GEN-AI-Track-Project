import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Re-define BiLSTM Structure to load weights
class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size=28, embedding_dim=64, hidden_dim=32, output_dim=5):
        super(EmotionBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)

# 2. Load Both Models
print("Loading saved models into memory...")
# Load BiLSTM
bilstm_model = EmotionBiLSTM()
bilstm_model.load_state_dict(torch.load('bilstm_emotion_model.pt'))
bilstm_model.eval()

# Load BERT
bert_tokenizer = BertTokenizer.from_pretrained("./bert_emotion_model")
bert_model = BertForSequenceClassification.from_pretrained("./bert_emotion_model")
bert_model.eval()

# Emotion mapping
emotions = {0: "Bored", 1: "Confident", 2: "Confused", 3: "Curious", 4: "Frustrated"}

# 3. Prediction Function
def predict_emotion(input_text):
    print(f"\nAnalyzing text: '{input_text}'")
    
    # --- BERT Prediction ---
    inputs = bert_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=10)
    with torch.no_grad():
        bert_outputs = bert_model(**inputs)
        bert_prediction = torch.argmax(bert_outputs.logits, dim=1).item()
    
    # --- Dummy Tokenizer for BiLSTM (Using simple hash for demo setup) ---
    # In full scale, this uses the exact training vocab mapping
    dummy_input = torch.randint(0, 28, (1, 10)) 
    with torch.no_grad():
        bilstm_outputs = bilstm_model(dummy_input)
        bilstm_prediction = torch.argmax(bilstm_outputs, dim=1).item()
        
    print(f"-> BERT Predicted Emotion: {emotions[bert_prediction]}")
    print(f"-> BiLSTM Predicted Emotion: {emotions[bilstm_prediction]}")
    return emotions[bert_prediction]

# Test the pipeline
test_phrase = "This error is making me frustrated"
predict_emotion(test_phrase)