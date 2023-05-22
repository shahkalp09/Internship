import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        self.df = pd.read_csv(file_path)
        self.max_seq_len = max_seq_len
        self.preprocess_data()

    def preprocess_data(self):
        positive_df = self.df[self.df['sentiment'] == 'positive'].head(150)
        negative_df = self.df[self.df['sentiment'] == 'negative'].head(150)
        df_reduced = pd.concat([positive_df, negative_df])
        df_reduced['review'] = df_reduced['review'].apply(lambda x: ' '.join(x.split()[:50]))
        df_reduced['review'] = df_reduced['review'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s\W]', '', x))
        df_reduced.reset_index(drop=True, inplace=True)
        self.data = df_reduced['review'].tolist()
        self.labels = df_reduced['sentiment'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        seq = [char_to_index[char] for char in text]
        seq += [char_to_index['<PAD>']] * (self.max_seq_len - len(seq))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# Load the dataset
file_path = "C:\\Users\\Admin\\OneDrive\\Desktop\\internship\\IMDB Dataset.csv\\IMDB Dataset.csv"
df = pd.read_csv(file_path)

# Create label encoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['sentiment'])

# Create vocabulary and character-to-index mapping
vocab = set()
max_seq_len = 0
for text in df['review']:
    vocab.update(text)
    max_seq_len = max(max_seq_len, len(text))
char_to_index = {char: i + 1 for i, char in enumerate(vocab)}
char_to_index['<PAD>'] = 0

# Create the dataset
dataset = IMDBDataset(file_path, max_seq_len)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, output_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Set the hyperparameters
vocab_size = len(vocab) + 1
embed_size = 100
num_filters = 100
filter_sizes = [3, 4, 5]
output_size = 1
learning_rate = 0.001
num_epochs = 10

model = CNN(vocab_size, embed_size, num_filters, filter_sizes, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

test_predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        predictions = torch.round(torch.sigmoid(outputs))
        test_predictions.extend(predictions.tolist())

test_labels = test_dataset.dataset.labels

from sklearn.metrics import classification_report
report = classification_report(test_labels, test_predictions)
print(report)
