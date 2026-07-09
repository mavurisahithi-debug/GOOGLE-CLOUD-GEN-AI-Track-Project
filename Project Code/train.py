import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 1. Sample Dataset Creation (For Testing)
data = {
    'text': [
        "I am so bored with this topic", 
        "I feel very confident about this exam", 
        "I am totally confused by this step", 
        "I am curious to learn more about AI", 
        "This error is making me frustrated"
    ],
    'label': [0, 1, 2, 3, 4]  # 0: Bored, 1: Confident, 2: Confused, 3: Curious, 4: Frustrated
}
df = pd.DataFrame(data)

# 2. Basic Tokenization (Converting Text to Numbers)
vocab = {"<PAD>": 0, "<UNK>": 1}
for text in df['text']:
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

def text_to_sequence(text, max_len=10):
    seq = [vocab.get(w, 1) for w in text.lower().split()]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    return seq[:max_len]

X = np.array([text_to_sequence(t) for t in df['text']])
y = df['label'].values

# 3. BiLSTM Model Architecture
class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(EmotionBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate forward and backward final hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)

# Model Parameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
OUTPUT_DIM = 5  # 5 Emotion Classes

model = EmotionBiLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
print(f"BiLSTM Model successfully created! Vocab Size: {VOCAB_SIZE}")

# Save Model Weights
torch.save(model.state_dict(), 'bilstm_emotion_model.pt')
print("bilstm_emotion_model.pt file saved successfully.")
