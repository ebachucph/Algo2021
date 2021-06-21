#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:32:55 2021

@author: algm
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_peptide_distribution(raw_data, raw_set):
    """
    needs BINDER_THRESHOLD defined in the main script
        
    Parameters
    ----------
    raw_data : pd array
        Data to be plotted
    raw_set : string
        string corresponding to the raw_data (i.e. train, test, etc.)

    Returns
    -------
    Plots the peptide distribution 
    

    """
    
    raw_data['peptide_length'] = raw_data.peptide.str.len()
    raw_data['target_binary'] = (raw_data.target >= BINDER_THRESHOLD).astype(int)

    # Position of bars on x-axis
    ind = np.arange(train_raw.peptide.str.len().nunique())
    neg = raw_data[raw_data.target_binary == 0].peptide_length.value_counts().sort_index()
    pos = raw_data[raw_data.target_binary == 1].peptide_length.value_counts().sort_index()

    # Plotting
    plt.figure()
    width = 0.3  

    plt.bar(ind, neg, width, label='Non-binders')
    plt.bar(ind + width, pos, width, label='Binders')

    plt.xlabel('Peptide lengths')
    plt.ylabel('Count of peptides')
    plt.title('Distribution of peptide lengths in %s data' %raw_set)
    plt.xticks(ind + width / 2, ['%dmer' %i for i in neg.index])
    plt.legend(loc='best')
    plt.show()


def plot_target_values(data, BINDER_THRESHOLD):
    plt.figure(figsize=(15,4))
    for partition, label in data:
        x = partition.index
        y = partition.target
        plt.scatter(x, y, label=label, marker='.')
    plt.axhline(y=BINDER_THRESHOLD, color='r', linestyle='--', label='Binder threshold')
    plt.legend(frameon=False)
    plt.title('Target values')
    plt.xlabel('Index of dependent variable')
    plt.ylabel('Dependent varible')
    plt.show()



def plot_roc_curve(fpr,tpr, roc_auc, peptide_length=[9]):
    """
    fpr, tpr needs to be defined in the main script 

    """
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f (%smer)' %(roc_auc, '-'.join([str(i) for i in peptide_length])))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], c='black', linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    


def plot_mcc(y_test, pred, mcc):
    plt.title('Matthews Correlation Coefficient')
    plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label = 'MCC = %0.2f' % mcc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Predicted')
    plt.xlabel('Validation targets')
    plt.show()
    

def performance_encoding_plot(df, perf_measure, errorbar):
    fig, axes = plt.subplots()
    for allele, d_ in df.groupby('Allele'):
        for encod, d in d_.groupby('Encoding'):
            #print(encod)
            
            axes.set_title("Performance versus Training set size for allele: %s"%allele)
            axes.errorbar(d["Train_size"].unique(), d[perf_measure], yerr=d[errorbar], linestyle='-', label=encod )
            axes.legend(loc = 'upper left')
            axes.set_ylabel('%s'%perf_measure)
            axes.set_xlabel("Training set size")
            
        fig.savefig("perf_enc_%s_%s"%(allele,perf_measure),dpi=200)
        plt.show()


def performance_testsize_boxplot(df, perf_measure, errorbar):
    
    # set width of bars
    barWidth = 0.25
    
    for allele, d_ in df.groupby('Allele'):
        
        # initialize plot
        fig, axes = plt.subplots()

        # initalize variables for the positions of the bars dynamically depending how many different
        # encoding schemes that we are using
        # the size needed is numb of encodings * barWidth + space between
        #r = [0+r*(barWidth*len(d_['Encoding'].unique())+2*barWidth) for r in np.arange(len(d_["Train_size"].unique()))]
        r = [0+r*(barWidth*len(d_['Encoding'].unique())+1*barWidth) for r in np.arange(len(d_["Train_size"].unique()))]
        
        for encod, d in d_.groupby('Encoding'):
            
            # Set position of bar on X axis
            # we loop over the different encodings and we need to update 
            # the position of the different bar posithin with + a barWidth 
            r = [x + barWidth for x in r]

            # plot the bars for the different encodings and test sizes
            axes.bar(r, d[perf_measure], width=barWidth, yerr=d[errorbar], label=encod)

            
        # make the plot look pretty
        axes.set_title("Performance versus Training set size for allele: %s"%allele, fontweight="bold", pad=10)
        axes.set_xlabel('Test size', fontweight='bold')
        axes.set_ylabel('%s'%perf_measure, fontweight='bold')
        # we are adjusting the x tick location from the last r (bar is in the middle) so we need to add
        # half a barWidth and then subtract the barWidth * numb of encoding so we move the tick to the middle
        axes.set_xticks([p+barWidth/2-(barWidth*len(d_['Encoding'].unique())/2) for p in r])
        axes.set_xticklabels(d_["Train_size"].unique())
        axes.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), fancybox=True, ncol=3)
        fig.tight_layout()
        
        out_dir=f"../data/{allele}_out"
        out_n=f"perf_testsize_box_{allele}_{perf_measure}-{'-'.join(d_['Encoding'].unique())}"
        fig.savefig(os.path.join(out_dir,out_n),dpi=200)
        #plt.show()

#performance_testsize_boxplot(df_test,"AUC","AUC_std")
#performance_testsize_boxplot(df_test,"MCC","MCC_std")
