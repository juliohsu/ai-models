import torch
import torch.nn as nn
import torch.nn.functional as F

# dataset 
# 
# ids for tokenize
vocab_dict = {
    "hello": 0,
    "world": 1,
    "how": 2,
    "are": 3,
    "you": 4,
    "good": 5,
    "morning": 6,
    "night": 7,
    "bye": 8,
}

# ids for destokenize
reverse_dict = {index: word for word, index in vocab_dict.items()}

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size=9, emb_size=32, hidden_size=128, max_seq_length=20):
        super(TextGeneratorModel, self).__init__()
        # embedding
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        # ltsm
        self.ltsm1 = nn.LSTM(
            input_size=emb_size, hidden_size=hidden_size, batch_first=True
        )
        self.ltsm2 = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )
        # activation
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.ltsm1(x)
        x, _ = self.ltsm2(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# activate model
model = TextGeneratorModel()

# training sentences
training_setences = [
    ["hello", "world"],
    ["how", "are", "you"],
    ["good", "morning"],
    ["good", "night"],
    ["bye"]
]

# training tokenize
tokenize_sentences = [[vocab_dict[word] for word in sentence] for sentence in training_setences]

# pad sequence for all sequences to be the max length
max_seq_length = max(len(seq) for seq in tokenize_sentences)

# sparse data, can be substitute by one-hot
for seq in tokenize_sentences:
    while len(seq) < max_seq_length:
        seq.append(0)

# training sentences
tokenize_sentences_tensor = torch.tensor(tokenize_sentences)

# hello world input
input_example = torch.tensor([[0, 1]])

# predict word
output_example = model(input_example)

# find the predicted word
most_prob_word_index = torch.argmax(output_example, dim=1).item()
word = reverse_dict[most_prob_word_index]

print(word)