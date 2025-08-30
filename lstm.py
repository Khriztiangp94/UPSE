import torch

class PyTorchLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, bidirectional=True):
        super(PyTorchLSTM, self).__init__()
        
        # Capa de embedding: convierte cada palabra en un vector denso
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        
        # LSTM: puede ser unidireccional o bidireccional
        self.lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Clasificador final
        direction = 2 if bidirectional else 1
        self.fc = torch.nn.Linear(hidden_dim * direction, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim * direction)
        out = lstm_out[:, -1, :]  # último estado oculto de la secuencia
        logits = self.fc(out)     # (batch, num_classes)
        return logits
