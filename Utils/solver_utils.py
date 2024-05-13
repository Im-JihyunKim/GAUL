import torch
from torch.autograd import Variable

def extract_source_features(n_sources, config, G, sx: list):
    Xs = sx.copy()
    del sx
    torch.cuda.empty_cache()
    for i in range(n_sources):
        for hidden in G:
            if hidden.__class__.__name__ == 'LSTM':
                Xs[i] = torch.transpose(Xs[i], -1, 1)
                (init_hidden, init_cell) = lstm_init_hidden(config, Xs[i].device, Xs[i].size(0))
                out, _ = hidden(Xs[i], (init_hidden, init_cell))
                Xs[i] = out[:,-1,:]
            else:
                Xs[i] = hidden(Xs[i])
    return Xs

def extract_target_features(config, G, tx: torch.FloatTensor):
    Xt = tx.clone()
    del tx
    torch.cuda.empty_cache()
    for hidden in G:
        if hidden.__class__.__name__ == 'LSTM':
            Xt = torch.transpose(Xt, -1, 1)
            (init_hidden, init_cell) = lstm_init_hidden(config, Xt.device, Xt.size(0))
            out, _ = hidden(Xt, (init_hidden, init_cell))
            Xt = out[:,-1,:]
        else:
            Xt = hidden(Xt)
    return Xt

def predictor1(n_sources, R1, sx: list, tx):
    y_spred = []
    if sx is not None:
        for i in range(n_sources):
            y_sx = sx[i].clone()
            for hidden in R1:
                y_sx = hidden(y_sx)
            y_spred.append(y_sx)

    y_tx = tx.clone()
    for hidden in R1:
        y_tx = hidden(y_tx)
    y_tpred = y_tx

    return y_spred, y_tpred

def predictor2(n_sources, R2, sx: list, tx):
    y_spred = []
    for i in range(n_sources):
        y_sx = sx[i].clone()
        for hidden in R2:
            y_sx = hidden(y_sx)
        y_spred.append(y_sx)

    y_tx = tx.clone()
    for hidden in R2:
        y_tx = hidden(y_tx)
    y_tpred = y_tx

    return y_spred, y_tpred

def lstm_init_hidden(config, device, seq_len: int):
    num_directions = 2*config.num_layers if config.bidirectional else 1*config.num_layers
    h_0 = Variable(torch.zeros(num_directions, seq_len, 32).to(device))
    c_0 = Variable(torch.zeros(num_directions, seq_len, 32).to(device))
    return h_0, c_0

def ema_update(ema_beta, pre_step_val, current_step_val):
    assert current_step_val is not None
    return ((1-ema_beta)*current_step_val + ema_beta*pre_step_val.data)