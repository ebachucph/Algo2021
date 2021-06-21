from parser import *
from encoding import *
from model import *
from plots import *
import os
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
MINI_BATCH_SIZE = 32
N_HIDDEN_NEURONS_1 = 64
N_HIDDEN_NEURONS_2 = 64
LEARNING_RATE = 0.01
PATIENCE = 10

## Loop over different encodings and training sizes
alleles = ['A0301', 'A0201', 'A3301']
encodings = ["MULTIPLE/BLOSUM50", "MULTIPLE/ONE_HOT", "MULTIPLE/ONE_HOT_FRAC", "MULTIPLE/ONE_HOT_MOD", ["SINGLE/CHARGE", "SINGLE/SIZE", "SINGLE/HYDROPHOB"], ]
#encodings = [["SINGLE/CHARGE", "SINGLE/SIZE", "SINGLE/HYDROPHOB"], "MULTIPLE/BLOSUM50"]

df_test = pd.DataFrame(columns=['Allele', 'Encoding', 'Train_size', 'MCC', 'MCC_bootstrap', 'MCC_std', 'AUC', 'AUC_bootstrap', 'AUC_std'])

n_kf_outer = 5
n_kf_inner = 5 

for j, allele in enumerate(alleles):
    ## Do parsing
    filename="../data/%s.dat"%allele
    X_raw = load_peptide_target(filename)
    
    # initialize an output directory using the allele name
    out_dir='../data/%s_out'%allele
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        

    for k, encoding in enumerate(encodings):

        ## Do encoding
        # scheme_file = f"../data/schemes/MULTIPLE/{encoding}"
        # X, y = encode_peptides(X_raw, scheme_file)
        X, y = encode_parser(X_raw, encoding)
        n_features = X.shape[-1]
        print(f"Number of features in encoding: {n_features}")
        train_sizes = [int(len(X)*((n_kf_outer-1)/n_kf_outer)*((n_kf_inner-1)/n_kf_inner)/5), 
                       int(len(X)*((n_kf_outer-1)/n_kf_outer)*((n_kf_inner-1)/n_kf_inner)/2), 
                       int(len(X)*((n_kf_outer-1)/n_kf_outer)*((n_kf_inner-1)/n_kf_inner))]
        
        
        for l, train_size in enumerate(train_sizes):

            pred_total = np.empty(0)
            y_test_total = np.empty(0)

            print(f"Training and testing on allele {allele} using encoding: {encoding} with training data size: {train_size}")           
            kf_outer_counter = 0

            # Extract test set
            kf_outer = KFold(n_splits=5)
            for train_index, test_index in kf_outer.split(X):
                X_trainval, X_test = X[train_index], X[test_index]
                y_trainval_, y_test_ = y[train_index], y[test_index]

                print(f"Test set {kf_outer_counter} with size: {len(X_test)}")

                # Split remaining data into random train and valid sets
                kf_inner = KFold(n_splits=5)
                pred_array = torch.empty(len(y_test_),0,dtype=torch.float32)

                for train_index, val_index in kf_inner.split(X_trainval):
                    X_train, X_val = X[train_index], X[val_index]
                    Y_train_, Y_val_ = y[train_index], y[val_index]

                    # Adjust training data size
                    x_train_ = X_train[:train_size]
                    y_train_ = Y_train_[:train_size]

                    # Reshape
                    x_train_ = x_train_.reshape(x_train_.shape[0], -1)
                    x_val_ = X_val.reshape(X_val.shape[0], -1)
                    x_test_ = X_test.reshape(X_test.shape[0], -1)
                    n_features = x_train_.shape[1]
                
                    # Convert to tensors
                    x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
                    y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)
                    x_val = Variable(torch.from_numpy(x_val_.astype('float32')))
                    y_val = Variable(torch.from_numpy(Y_val_.astype('float32'))).view(-1, 1)
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
                    net, _, _ = train_with_minibatches(net, train_loader, valid_loader, EPOCHS, PATIENCE, optimizer, criterion)
                
                    ## Evaluate on test set
                    # Make test predictions
                    net.eval()
                    pred = net(x_test)
                    pred_array = torch.cat((pred_array, pred), axis=1)
      
                # Take mean of inner loop predictions
                pred_mean = torch.mean(pred_array, 1)   
                kf_outer_counter += 1    
 
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
                boot = resample(test_data, replace=True, n_samples=int(len(test_data)/5), random_state=i)
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
            
            # make a new variable that is just the name of the encoding scheme if we are not combining
            # else it is a concatenated name consisting of the different encoding schemes names
            if type(encoding) == list:
                tmp_ = [x.split('/')[1] for x in encoding]
                encoding_ = "_".join(tmp_)
            else:
                encoding_ = encoding.split("/")[1]

            # Add to df
            df_test = df_test.append({'Allele': allele, 
                                      'Encoding': encoding_, 
                                      'Train_size': train_size, 
                                      'MCC': mcc_mean,
                                      'MCC_std': mcc_std,
                                      'MCC_bootstrap' : mcc_list,
                                      'AUC': auc_mean,
                                      'AUC_std': auc_std,
                                      'AUC_bootstrap': auc_list,
                                       }, 
                                       ignore_index = True)
            print(f"Allele {allele} using encoding {encoding} with training size {train_size} achieves MCC of {mcc_mean:.2f} and AUC of {auc_mean:.2f}")
            out_n=f"{allele}-{'-'.join(df_test['Encoding'].unique())}.csv"
            df_test.to_csv(os.path.join(out_dir,out_n))
            
# Do plotting
performance_encoding_plot(df_test,"MCC","MCC_std")
performance_encoding_plot(df_test,"AUC","AUC_std")
performance_testsize_barplot(df_test,"MCC","MCC_std")
performance_testsize_barplot(df_test,"AUC","AUC_std")
boxplot(df_test, 'A0201', 1976, 'AUC_bootstrap')
boxplot(df_test, 'A0201', 1976, 'MCC_bootstrap')
#barplot_oneallele(df_test,'A0201', 1976, "AUC","AUC_std")
