import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# Read the CSV file
df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\internship\\language.csv")

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])

# Create a vocabulary of unique characters in the 'Name' column
char_vocab = set(''.join(df['Name']))
char_to_index = {char: i for i, char in enumerate(char_vocab)}
vocab_size = len(char_vocab)
print(vocab_size)

# Function to convert text to character embeddings
def text_to_char_embeddings(text):
    char_embeddings = [char_to_index[char] for char in text]
    return torch.tensor(char_embeddings, dtype=torch.float32)

# CNN model class
class CNN(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=2)
        self.fc = nn.Linear(256, num_classes)
        self.vocab_size = vocab_size

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x

# Initialize the CNN model
num_classes = len(label_encoder.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(num_classes, vocab_size).to(device)

# Load the trained model
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Function to predict the category of a name
def predict_language(name):
    with torch.no_grad():
        name_embeddings = text_to_char_embeddings(name)
        name_input = name_embeddings.unsqueeze(0).to(device)
        output = model(name_input).squeeze(0)
        _, predicted_class = torch.max(output, 0)
        predicted_language = label_encoder.classes_[predicted_class.item()]
        return predicted_language

# Example usage

print(predict_language(input_name))
