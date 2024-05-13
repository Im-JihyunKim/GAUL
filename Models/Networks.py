import torch.nn as nn

def feature_extractor(input_dim:int, bidirectional:bool, num_layers:int=2):
    G = nn.ModuleList([
            nn.Conv1d(input_dim, 64, kernel_size=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.15),

            nn.Conv1d(64, 30, kernel_size=16),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.15),

            nn.LSTM(input_size=30, hidden_size=32, batch_first=True, num_layers=num_layers,
                    bidirectional=bidirectional),
            ])
    
    return G

def predictor1(bidirectional:bool):
    R1 = nn.ModuleList([
        nn.Linear(64, 32) if bidirectional else nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(32, 1) if bidirectional else nn.Linear(16, 1),
    ])
    return R1

def predictor2(bidirectional:bool):
    R2 = nn.ModuleList([
        nn.Linear(64, 32) if bidirectional else nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(32, 1) if bidirectional else nn.Linear(16, 1),
    ])
    return R2