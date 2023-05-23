import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\internship\\language.csv")

print(df.head())
print(df['Category'].value_counts())

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Read the dataset from the CSV file
#df = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/internship/language.csv")

# Define the categories and their desired sample count
category_counts = {
    'russian': 200,
    'english': 200,
    'arabic': 200,
    'czech': 200,
    'japanese': 200,
    'german': 200,
    'italian': 200,
    'spanish': 200,
    'dutch': 200,
    'french': 200,
    'chinese': 200,
    'irish': 200,
    'greek': 200,
    'polish': 200,
    'scottish': 200,
    'korean': 200,
    'portugese': 200,
    'vietnamese': 200
}

df_balanced = pd.DataFrame()
for category, count in category_counts.items():
    df_category = df[df['Category'] == category]
    if len(df_category) > count:
        df_category_sampled = df_category.sample(n=count, random_state=42)
    else:
        df_category_sampled = df_category.sample(n=count, replace=True, random_state=42)
    df_balanced = pd.concat([df_balanced, df_category_sampled])

df_balanced = df_balanced.reset_index(drop=True)
print(df_balanced['Category'].value_counts())
print(df_balanced[df_balanced['Category']=='english'])
print(df_balanced.shape)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
torch.manual_seed(42)
np.random.seed(42)
label_encoder = LabelEncoder()
df_balanced['label'] = label_encoder.fit_transform(df_balanced['Category'])
char_vocab = set(''.join(df_balanced['Name']))
char_to_index = {char: i for i, char in enumerate(char_vocab)}
vocab_size = len(char_vocab)
print(vocab_size)
def text_to_char_embeddings(text):
    char_embeddings = [char_to_index[char] for char in text]
    return torch.tensor(char_embeddings, dtype=torch.float32)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = self.X.iloc[index]
        label = self.y.iloc[index]
        return text, label
dataset = CustomDataset(df_balanced['Name'], df_balanced['label'])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
def collate_fn(data):
    inputs, targets = zip(*data)
    inputs = [text_to_char_embeddings(text) for text in inputs]
    inputs = pad_sequence(inputs, batch_first=True)
    targets = torch.tensor(targets)
    return inputs, targets
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
class CNN(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3)
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Train the model
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).long()  # Convert targets to torch.long

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)  # Squeeze the output tensor
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    # Calculate average training loss
    train_loss /= len(train_loader.dataset)

    # Evaluate on the test set
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()  # Convert targets to torch.long

            outputs = model(inputs).squeeze(1)  # Squeeze the output tensor
            _, predicted = torch.max(outputs, 1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate evaluation metrics
    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average='macro')
    test_recall = recall_score(y_true, y_pred, average='macro')
    test_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, "
          f"Test Accuracy: {test_accuracy:.4f}, "
          f"Test Precision: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, "
          f"Test F1-Score: {test_f1:.4f}")
