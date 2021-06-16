from parser import *
from encoding import *
from model import *
from plots import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

## Arguments
BINDER_THRESHOLD = 0.426

## Call parser
allele='A0301'
filename="../data/%s.dat"%allele
X_raw = load_peptide_target(filename)

## Call encoder
scheme_file = "../data/schemes/BLOSUM50" 
X_encoded, Y = encode_peptides(X_raw, scheme_file) 

## Split into train and test
TRAIN_VALID_SPLIT = int(0.8*len(X_encoded))

x_train_ = X_encoded[:TRAIN_VALID_SPLIT,:,:]
x_valid_ = X_encoded[TRAIN_VALID_SPLIT:,:]
y_train_ = Y[:TRAIN_VALID_SPLIT]
y_valid_ = Y[TRAIN_VALID_SPLIT:]

x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
n_features = x_train_.shape[1]

x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)
x_valid = Variable(torch.from_numpy(x_valid_.astype('float32')))
y_valid = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)

## Train ANN
# Define hyperparameters
EPOCHS = 1000
MINI_BATCH_SIZE = 16
N_HIDDEN_NEURONS_1 = 32
N_HIDDEN_NEURONS_2 = 32
LEARNING_RATE = 0.01
PATIENCE = EPOCHS // 10

# Initialize net
net = ANN(n_features, N_HIDDEN_NEURONS_1, N_HIDDEN_NEURONS_2)
net.apply(init_weights)
net

# Initialize optimizer and loss
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Train ANN
train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=MINI_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=TensorDataset(x_valid, y_valid), batch_size=MINI_BATCH_SIZE, shuffle=True)
net, train_loss, valid_loss = train_with_minibatches(net, train_loader, valid_loader, EPOCHS, PATIENCE, optimizer, criterion)

# Plot learning curve
plot_losses(train_loss, valid_loss)

## Evaluation

# evaluate on test set
net.eval()
pred = net(x_valid)
loss = criterion(pred, y_valid)


# plot target values
plot_target_values([(pd.DataFrame(pred.data.numpy(), columns=['target']), 'Prediction'), (X_raw.iloc[TRAIN_VALID_SPLIT:].reset_index(), 'Target')], BINDER_THRESHOLD)


# transform targets to classes
y_test_class = np.where(y_valid.flatten() >= BINDER_THRESHOLD, 1, 0)
y_pred_class = np.where(pred.flatten() >= BINDER_THRESHOLD, 1, 0)


# Plot the ROC

