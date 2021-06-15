import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def init_weights(m):
    """
    https://pytorch.org/docs/master/nn.init.html
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01) # alternative command: m.bias.data.fill_(0.01)

class ANN(nn.Module):
    def __init__(self, n_features, n_l1, n_l2):
        super(ANN, self).__init__()
        # Computational layers
        self.fc1 = nn.Linear(n_features, n_l1)
        self.fc2 = nn.Linear(n_l1, n_l2)
        self.fc3 = nn.Linear(n_l2, 1)
        
        # BatchNorm
        self.bn_1 = nn.BatchNorm1d(num_features=n_l1)
        self.bn_2 = nn.BatchNorm1d(num_features=n_l2)

    def forward(self, x):
        x = self.bn_1(nn.LeakyReLU(self.fc1(x)))
        x = self.bn_2(nn.LeakyReLU(self.fc2(x)))
        x = self.fc3(x)
        return x
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping invoked!")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
    
def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True


def train_with_minibatches():
    
    train_loss, valid_loss = [], []

    early_stopping = EarlyStopping(patience=PATIENCE)
    for epoch in range(EPOCHS):
        batch_loss = 0
        net.train()
        for x_train, y_train in train_loader:
            pred = net(x_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.data
        train_loss.append(batch_loss / len(train_loader))

        if epoch % (EPOCHS//10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss[-1]))

        batch_loss = 0
        net.eval()
        for x_valid, y_valid in valid_loader:
            pred = net(x_valid)
            loss = criterion(pred, y_valid)
            batch_loss += loss.data
        valid_loss.append(batch_loss / len(valid_loader))

        if invoke(early_stopping, valid_loss[-1], net, implement=True):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break
            
    return net, train_loss, valid_loss


def plot_losses(burn_in=20):
    plt.figure(figsize=(15,4))
    plt.plot(list(range(burn_in, len(train_loss))), train_loss[burn_in:], label='Training loss')
    plt.plot(list(range(burn_in, len(valid_loss))), valid_loss[burn_in:], label='Validation loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Minimum Validation Loss')

    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()