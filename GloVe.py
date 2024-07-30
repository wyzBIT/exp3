import os
from collections import Counter
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 配置参数
config = {
    'data_dir': 'data/imdb/aclImdb',
    'embedding_file': 'glove.6B/glove.6B.100d.txt',  # 预训练词嵌入文件路径
    'batch_size': 32,
    'embedding_dim': 100,
    'hidden_dim': 256,
    'output_dim': 2,
    'vocab_size': 10000,
    'max_len': 100,
    'dropout_prob': 0.5,
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'epochs': 30,
    'pad_token': '<pad>',
    'unk_token': '<unk>',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

print(f"Using device: {config['device']}")

# 读取数据
def read_data(directory):
    print(f"Reading data from {directory}...")
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(directory, label_type)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
        for fname in tqdm(os.listdir(dir_path), desc=f"Reading {label_type} files"):
            with open(os.path.join(dir_path, fname), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    print(f"Data reading completed. Number of texts: {len(texts)}")
    return texts, labels

train_texts, train_labels = read_data(os.path.join(config['data_dir'], 'train'))
test_texts, test_labels = read_data(os.path.join(config['data_dir'], 'test'))

# 数据预处理
def preprocess_text(text):
    text = text.lower()
    text = text.replace('br', '')
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return tokens

print("Preprocessing text data...")
train_texts = [preprocess_text(text) for text in tqdm(train_texts, desc="Processing train texts")]
test_texts = [preprocess_text(text) for text in tqdm(test_texts, desc="Processing test texts")]

# 构建词汇表
def build_vocab(texts, top_n=None):
    counter = Counter()
    for text in texts:
        counter.update(text)
    common_words = counter.most_common(top_n)
    return Counter(dict(common_words))

print("Building vocabulary...")
vocab = build_vocab(train_texts, config['vocab_size'])
vocab_size = len(vocab)
word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}
word_to_idx[config['pad_token']] = 0
word_to_idx[config['unk_token']] = len(word_to_idx)
pad_idx = word_to_idx[config['pad_token']]

# 加载预训练的词向量
def load_pretrained_embeddings(embedding_file, word_to_idx, embedding_dim):
    embeddings = np.zeros((len(word_to_idx), embedding_dim))
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if word in word_to_idx:
                idx = word_to_idx[word]
                embeddings[idx] = vector
    return torch.tensor(embeddings, dtype=torch.float32)

print("Loading pretrained embeddings...")
pretrained_embeddings = load_pretrained_embeddings(config['embedding_file'], word_to_idx, config['embedding_dim'])

# 将文本转换为索引
def texts_to_indices(texts, word_to_idx, max_len):
    indices = []
    for text in texts:
        text_indices = [word_to_idx.get(word, word_to_idx[config['unk_token']]) for word in text]
        if len(text_indices) < max_len:
            text_indices += [pad_idx] * (max_len - len(text_indices))
        else:
            text_indices = text_indices[:max_len]
        indices.append(text_indices)
    return indices

print("Converting texts to indices...")
train_indices = texts_to_indices(train_texts, word_to_idx, config['max_len'])
test_indices = texts_to_indices(test_texts, word_to_idx, config['max_len'])

# IMDB数据集类
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

print("Creating datasets and dataloaders...")
train_dataset = IMDBDataset(train_indices, train_labels)
test_dataset = IMDBDataset(test_indices, test_labels)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

# LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(pretrained_embeddings)  # 使用预训练的词嵌入
        self.embedding.weight.requires_grad = False  # 固定预训练词嵌入
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Apply dropout after LSTM layer
        x = self.fc(x)
        return x


# 训练模型的函数
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(config['device']), labels.to(config['device'])
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).sum().item() / len(labels)
        epoch_acc += acc

    avg_loss = epoch_loss / len(train_loader)
    avg_acc = epoch_acc / len(train_loader)

    return avg_loss, avg_acc

# 测试模型的函数
def test_epoch(model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    correct_pred = [0] * config['output_dim']
    total_pred = [0] * config['output_dim']

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(config['device']), labels.to(config['device'])
            outputs = model(texts)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).cpu().numpy()

            for i in range(len(labels)):
                total_pred[labels[i].item()] += 1
                if correct[i]:
                    correct_pred[labels[i].item()] += 1

            acc = (predicted == labels).sum().item() / len(labels)
            epoch_acc += acc

    avg_loss = epoch_loss / len(test_loader)
    avg_acc = epoch_acc / len(test_loader)

    class_accuracies = [correct_pred[i] / total_pred[i] if total_pred[i] > 0 else 0 for i in
                        range(config['output_dim'])]

    return avg_loss, avg_acc, class_accuracies

# 训练和测试 LSTM 模型
print("Training LSTM model...")
lstm_model = LSTMModel(vocab_size=vocab_size + 2,
                       embedding_dim=config['embedding_dim'],
                       hidden_dim=config['hidden_dim'],
                       output_dim=config['output_dim'],
                       pad_idx=pad_idx,
                       dropout_prob=config['dropout_prob']).to(config['device'])
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

lstm_train_losses, lstm_train_accuracies = [], []
lstm_test_losses, lstm_test_accuracies = [], []
lstm_class_accuracies = []

for epoch in range(config['epochs']):
    print(f"\nLSTM Epoch {epoch + 1}/{config['epochs']}", end="\t")

    train_loss, train_acc = train_epoch(lstm_model, train_loader, lstm_criterion, lstm_optimizer)
    test_loss, test_acc, class_acc = test_epoch(lstm_model, test_loader, lstm_criterion)

    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f},', end=" ")
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}', end=" ")

    lstm_train_losses.append(train_loss)
    lstm_train_accuracies.append(train_acc)
    lstm_test_losses.append(test_loss)
    lstm_test_accuracies.append(test_acc)
    lstm_class_accuracies.append(class_acc)


# 输出各模型各类别准确率和总准确率
print("\nFinal Epoch Metrics:")
print("\nLSTM Model:")
for i, acc in enumerate(lstm_class_accuracies[-1]):
    print(f"LSTM - Class {i} Accuracy: {acc:.4f}")
print(f"LSTM - Overall Test Accuracy: {lstm_test_accuracies[-1]:.4f}")

