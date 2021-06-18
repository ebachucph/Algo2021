from parser import *
from encoding import *
from model import *
from plots import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.utils import resample
import pandas as pd

# Define hyperparameters
EPOCHS = 1000
MINI_BATCH_SIZE = 64
N_HIDDEN_NEURONS_1 = 32
N_HIDDEN_NEURONS_2 = 32
LEARNING_RATE = 0.01
PATIENCE = 10

## Loop over different encodings and training sizes
alleles = ['A0301']
encodings = ["BLOSUM50","ONE_HOT", "ONE_HOT_MOD"]
train_sizes = [200, 500, 1000]
perm_test = len(alleles)*len(encodings)*len(train_sizes)

df_test = pd.DataFrame(columns=['Allele', 'Encoding', 'Train_size', 'MCC', 'MCC_std', 'AUC', 'AUC_std'])

pred_total = np.empty(0)
y_test_total = np.empty(0)

for j, allele in enumerate(alleles):
    ## Do parsing
    filename="../data/%s.dat"%allele
    X_raw = load_peptide_target(filename)

    for k, encoding in enumerate(encodings):
        for l, train_size in enumerate(train_sizes):
       
            print(f"Training and testing on allele {allele} using encoding: {encoding} with training data size: {train_size}") 
    
            ## Do encoding
            scheme_file = f"../data/schemes/{encoding}"
            X, y = encode_peptides(X_raw, scheme_file)
            n_features = X.shape[-1]
            print(f"Number of features in encoding: {n_features}")
    
            # Extract test set
            kf_outer = KFold(n_splits=5)
            for train_index, test_index in kf_outer.split(X):
                X_trainval, X_test = X[train_index], X[test_index]
                y_trainval_, y_test_ = y[train_index], y[test_index]
  
                print(f"Test set size: {len(X_test)}")

                # Split remaining data into random train and valid sets
                kf_inner = KFold(n_splits=5)
                pred_array = torch.empty(len(y_test_),0,dtype=torch.float32)

                for train_index, val_index in kf_inner.split(X_trainval):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train_, y_val_ = y[train_index], y[val_index]

                    # Adjust training data size
                    x_train_ = X_train[:train_size]
                    y_train_ = y_train_[:train_size]

                    # Reshape
                    x_train_ = x_train_.reshape(x_train_.shape[0], -1)
                    x_val_ = X_val.reshape(X_val.shape[0], -1)
                    x_test_ = X_test.reshape(X_test.shape[0], -1)
                    n_features = x_train_.shape[1]
            
                    # Convert to tensors
                    x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
                    y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)
                    x_val = Variable(torch.from_numpy(x_val_.astype('float32')))
                    y_val = Variable(torch.from_numpy(y_val_.astype('float32'))).view(-1, 1)
                    x_test = Variable(torch.from_numpy(x_test_.astype('float32')))
                    y_test = Variable(torch.from_numpy(y_test_.astype('float32'))).view(-1, 1)
            
                    # Initialize net
                    net = ANN(n_features, N_HIDDEN_NEURONS_1, N_HIDDEN_NEURONS_2)
                    net.apply(init_weights)
            
                    # Initialize optimizer and loss
                    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
                    criterion = nn.MSELoss()
                    
                    # Train ANN
                    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=MINI_BATCH_SIZE, shuffle=True)
                    valid_loader = DataLoader(dataset=TensorDataset(x_val, y_val), batch_size=MINI_BATCH_SIZE, shuffle=True)
                    net, train_loss, valid_loss = train_with_minibatches(net, train_loader, valid_loader, EPOCHS, PATIENCE, optimizer, criterion)
            
                    ## Evaluate on test set
                    # Make test predictions
                    net.eval()
                    pred = net(x_test)
                    pred_array = torch.cat((pred_array, pred), axis=1)
  
                # Take mean of inner loop predictions
                pred_mean = torch.mean(pred_array, 1)   
 
                # Save test set and predictions
                pred_outerfold = pred_mean.detach().numpy().reshape(-1)
                y_test = y_test.detach().numpy().reshape(-1)
                pred_total = np.append(pred_total, pred_outerfold, axis=0)
                y_test_total = np.append(y_test_total, y_test, axis=0)

            # Do bootstrapping:
            n_boot = 1000
            test_data = np.concatenate((pred_total.reshape(-1,1), y_test_total.reshape(-1,1)), axis=1)        
            mcc_list = np.zeros(n_boot)
            auc_list = np.zeros(n_boot)
 
            for i in range(n_boot):  
                boot = resample(test_data, replace=True, n_samples=500, random_state=i)
                pred_boot = boot[:,0]
                y_test_boot = boot[:,1] 

                # Transform test data and predictions into classes
                BINDER_THRESHOLD = 0.426
                y_test_class = np.where(y_test_boot.flatten() >= BINDER_THRESHOLD, 1, 0)
                y_pred_class = np.where(pred_boot.flatten() >= BINDER_THRESHOLD, 1, 0)
        
                # Compute correlations
                mcc = matthews_corrcoef(y_test_class, y_pred_class)
                mcc_list[i] = mcc              

                # Compute AUC
                fpr, tpr, threshold = roc_curve(y_test_class.reshape(-1), pred_boot.reshape(-1))
                roc_auc = auc(fpr, tpr)
                auc_list[i] = roc_auc           

            # Compute error bars
            mcc_mean = np.mean(mcc_list)
            mcc_std = np.std(mcc_list)
            auc_mean = np.mean(auc_list)    
            auc_std = np.std(auc_list)

            # Add to df
            df_test = df_test.append({'Allele': allele, 
                                      'Encoding': encoding, 
                                      'Train_size': train_size, 
                                      'MCC': mcc_mean,
                                      'MCC_std': mcc_std,
                                      'AUC': auc_mean,
                                      'AUC_std': auc_std,
                                       }, 
                                       ignore_index = True)
            print(f"Allele {allele} using encoding {encoding} with training size {train_size} achieves MCC of {mcc_mean:.2f} and AUC of {auc_mean:.2f}")

# Do plotting
performance_encoding_plot(df_test,"MCC","MCC_std")
performance_encoding_plot(df_test,"AUC","AUC_std")
