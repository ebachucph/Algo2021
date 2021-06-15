from model import *
from parser import *

## Call parser
allele='A0301'
filename="../data/%s.dat"%allele

raw_data=load_peptide_target(filename)


## Call encoder

## Train ANN
# Define hyperparameters
EPOCHS = 1000
MINI_BATCH_SIZE = 16
N_HIDDEN_NEURONS_1 = 32
N_HIDDEN_NEURONS_2 = 32
LEARNING_RATE = 0.01
PATIENCE = EPOCHS // 10

# Initialize model
model = ANN(n_features, N_HIDDEN_NEURONS_1, N_HIDDEN_NEURONS_2)
model.apply(init_weights)
net

# Initialize optimizer and loss
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Train ANN
train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=MINI_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=TensorDataset(x_valid, y_valid), batch_size=MINI_BATCH_SIZE, shuffle=True)
net, train_loss, valid_loss = train_with_minibatches()

# Plot learning curve
plot_losses(train_loss, valid_loss)


