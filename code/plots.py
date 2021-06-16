#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:32:55 2021

@author: algm
"""

import matplotlib.pyplot as plt
import pandas as pd

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
    

def performance_encoding_plot(df, perf_meassure):
    print(type(perf_meassure))
    for allele, d_ in df.groupby('alleles'):
        #print(d_)
        fig, axes = plt.subplots()
        for encod, d in d_.groupby('encoding'):
            axes.set_title("Performance versus Training Set Size %s"%allele)
            axes.plot(d.sizes, d[perf_meassure], linestyle='-', label=encod )
            axes.legend(loc = 'upper left')
            #plt.plot([0, 1], [0, 1], c='black', linestyle='--')
            axes.set_ylabel('%s'%perf_meassure)
            axes.set_xlabel("Training Set Size")
        fig.savefig("perf_enc_%s"%allele)
        plt.show()
    
'''
alleles=["A0101"]*8+["A0201"]*8
encoding=(["1-hot"]*4+["sparse"]*4)*2
sizes= [10,20,30,40]*4
auc = ([0.1,0.2,0.15,0.11]+[0.3,0.6,0.25,0.71])*2
mcc = ([0.1,0.4,0.15,0.11]+[0.2,0.6,0.25,0.71])*2

df=pd.DataFrame([alleles,encoding,sizes,auc,mcc], index=["alleles","encoding","sizes","auc","mcc"]).T

performance_encoding_plot(df, 'auc')
performance_encoding_plot(df, 'mcc')
'''
    