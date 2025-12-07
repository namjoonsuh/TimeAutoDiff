import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

class PosthocRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PosthocRNN, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim-1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def predictive_score_metrics(ori_data, generated_data, col_pred, iterations=5000, batch_size=128):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)
    dim = ori_data.shape[2]
    hidden_dim = int(dim/2)
    model = PosthocRNN(dim, hidden_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    for _ in tqdm(range(iterations), desc="Training"):
        model.train()
        idx = np.random.permutation(len(generated_data))[:batch_size]       
        X_train = torch.cat((generated_data[idx, :-1, :col_pred], generated_data[idx, :-1, col_pred + 1:]), dim=2)
        Y_train = generated_data[idx, 1:, col_pred].unsqueeze(-1)
        
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    MAE_temp = 0
    with torch.no_grad():
        for i in range(len(ori_data)):
            X_test = torch.cat((ori_data[i:i+1, :-1, :col_pred], ori_data[i:i+1, :-1, col_pred + 1:]), dim=2)
            Y_test = ori_data[i:i+1, 1:, col_pred].unsqueeze(-1)
            prediction = model(X_test)
            MAE_temp += mean_absolute_error(Y_test.cpu().squeeze(2).numpy(), prediction.cpu().squeeze(2).numpy())

    predictive_score = MAE_temp / len(ori_data)

    return predictive_score
