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

## Do parsing
allele='A0301'
filename="../data/%s.dat"%allele
X_raw = load_peptide_target(filename)

# Define hyperparameters
EPOCHS = 1000
MINI_BATCH_SIZE = 32
N_HIDDEN_NEURONS_1 = 16
N_HIDDEN_NEURONS_2 = 16
LEARNING_RATE = 0.01
PATIENCE = EPOCHS // 10

## Loop over different encodings and training sizes
encodings = ["BLOSUM50","ONE_HOT"]
train_sizes = [500, 1000]

corr_mat = np.zeros((len(encodings), len(encodings)))
auc_mat = np.zeros((len(encodings), len(encodings)))

for i, encoding in enumerate(encodings):
    for j, train_size in enumerate(train_sizes):
       
        print(f"Training and testing using encoding: {encoding} with training data size: {train_size}")    

        ## Do encoding
        scheme_file = f"../data/schemes/{encoding}"
        X_encoded, Y = encode_peptides(X_raw, scheme_file)
        n_features = X_encoded.shape[-1]
        print(f"Number of features in encoding: {n_features}")

		## Prepare data for ANN
		#Split into train, validation and test
        TEST_SPLIT = 0.1
        TRAIN_VALID_SPLIT = 0.8
        X_encoded_test = X_encoded[:int(TEST_SPLIT*len(X_encoded))]
        Y_test = Y[:int(TEST_SPLIT*len(Y))]
        x_test_ = X_encoded_test
        y_test_ = Y_test

        # Extract train and valid data
        X_encoded_trainval = X_encoded[int(TEST_SPLIT*len(X_encoded)):]
        Y_trainval = Y[int(TEST_SPLIT*len(X_encoded)):]

		# Split into train and val
        x_train_ = X_encoded_trainval[:int(TRAIN_VALID_SPLIT*len(X_encoded_trainval))]
        x_valid_ = X_encoded_trainval[int(TRAIN_VALID_SPLIT*len(X_encoded_trainval)):]
        y_train_ = Y_trainval[:int(TRAIN_VALID_SPLIT*len(X_encoded_trainval))]
        y_valid_ = Y_trainval[int(TRAIN_VALID_SPLIT*len(X_encoded_trainval)):]

        # Adjust training data size
        #print(len(x_train_))
        x_train_ = x_train_[:train_size]
        y_train_ = y_train_[:train_size]

        # Reshape
        x_train_ = x_train_.reshape(x_train_.shape[0], -1)
        x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
        x_test_ = x_test_.reshape(x_test_.shape[0], -1)
        n_features = x_train_.shape[1]

        # Convert to tensors
        x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
        y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)
        x_valid = Variable(torch.from_numpy(x_valid_.astype('float32')))
        y_valid = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)
        x_test = Variable(torch.from_numpy(x_valid_.astype('float32')))
        y_test = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)

        # Initialize net
        net = ANN(n_features, N_HIDDEN_NEURONS_1, N_HIDDEN_NEURONS_2)
        net.apply(init_weights)

        # Initialize optimizer and loss
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
 
        # Train ANN
        train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=MINI_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(dataset=TensorDataset(x_valid, y_valid), batch_size=MINI_BATCH_SIZE, shuffle=True)
        net, train_loss, valid_loss = train_with_minibatches(net, train_loader, valid_loader, EPOCHS, PATIENCE, optimizer, criterion)

        # Plot learning curve
        #plot_losses(train_loss, valid_loss)

        ## Evaluate on test set
        # Make test predictions
        net.eval()
        pred = net(x_test)

        # Transform test data and predictions into classes
        BINDER_THRESHOLD = 0.426
        y_test_class = np.where(y_test.flatten() >= BINDER_THRESHOLD, 1, 0)
        y_pred_class = np.where(pred.flatten() >= BINDER_THRESHOLD, 1, 0)

        # Compute correlations
        mcc = matthews_corrcoef(y_test_class, y_pred_class)
        corr_mat[i,j] = mcc

        # Compute AUC
        fpr, tpr, threshold = roc_curve(y_test_class, pred.flatten().detach().numpy())
        roc_auc = auc(fpr, tpr)
        auc_mat[i,j] = roc_auc
 
        print(f"Encoding {encoding} with training size {train_size} achieves MCC of {mcc:.2f} and AUC {roc_auc:.2f}")

print(corr_mat)
print(auc_mat)
