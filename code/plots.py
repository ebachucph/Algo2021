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
    #fig, axes = plt.subplots()
    for allele, d_ in df.groupby('Allele'):
        fig, axes = plt.subplots()
        for encod, d in d_.groupby('Encoding'):
            #print(encod)
            
            axes.set_title("Performance versus Training set size for allele: %s"%allele, pad=10)
            axes.errorbar(d["Train_size"].unique(), d[perf_measure], yerr=d[errorbar], linestyle='-', label=encod, capsize=3, alpha=0.8 )
            axes.legend(loc = 'lower right')
            axes.set_ylabel('%s'%perf_measure)
            #axes.set_xlabel("Training set size")
            axes.set_xlabel("fraction of training set")
            if perf_measure == 'AUC':
                axes.set_ylim([0.70,1])
            if perf_measure == 'MCC':
                axes.set_ylim([0,0.9])
            axes.set_xticks([int(t) for t in d_["Train_size"].unique()])
            #axes.set_xticklabels(d_["Train_size"].unique())
            axes.set_xticklabels([0.2,0.5,1])
        
        
        out_dir=f"../data/{allele}_out"
        out_n=f"perf_testsize_line_{allele}_{perf_measure}-{'-'.join(d_['Encoding'].unique())}"
        fig.savefig(os.path.join(out_dir,out_n),dpi=200)
        #fig.savefig("perf_enc_%s_%s"%(allele,perf_measure),dpi=200)
        #plt.show()


def performance_testsize_barplot(df, perf_measure, errorbar):
    
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
            axes.bar(r, d[perf_measure], width=barWidth, yerr=d[errorbar], label=encod, capsize=3, alpha=0.8, ecolor='darkslategray')

            
        # make the plot look pretty
        axes.set_title("Performance versus Training set size for allele: %s"%allele, pad=10)
        #axes.set_xlabel('Training set size')
        axes.set_xlabel("fraction of training set")
        axes.set_ylabel('%s'%perf_measure)
        if perf_measure == 'AUC':
            axes.set_ylim([0.70,1])
        if perf_measure == 'MCC':
            axes.set_ylim([0,0.9])
        # we are adjusting the x tick location from the last r (bar is in the middle) so we need to add
        # half a barWidth and then subtract the barWidth * numb of encoding so we move the tick to the middle
        axes.set_xticks([p+barWidth/2-(barWidth*len(d_['Encoding'].unique())/2) for p in r])
        #axes.set_xticklabels(d_["Train_size"].unique())
        axes.set_xticklabels([0.2,0.5,1])
        axes.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), fancybox=True, ncol=3)
        fig.tight_layout()
        
        out_dir=f"../data/{allele}_out"
        out_n=f"perf_testsize_box_{allele}_{perf_measure}-{'-'.join(d_['Encoding'].unique())}"
        fig.savefig(os.path.join(out_dir,out_n),dpi=200)
        #plt.show()


def boxplot(df, allele, train_size, bootstrap_measure):
    
    # set width of bars
    barWidth = 0.25
    
    #for allele, d_ in df.groupby('Allele'):
        
    # initialize plot
    fig, axes = plt.subplots()
    
    d_ = df.loc[df.Allele == allele].loc[df.Train_size == train_size]
    
    
    #axes.boxplot(d_[bootstrap_measure], labels=d_['Encoding'].unique())

    pos=0
    data=np.array(len)
    labels=[]
    for encod, d in d_.groupby('Encoding'):
        #print(d)
        labels.append(encod)
        axes.boxplot(d[bootstrap_measure], positions=[pos], widths=[0.5])
        pos+=0.75
        
    axes.set_title("Performance for allele: %s"%allele, pad=10)
    axes.set_xlabel("Encoding scheme", labelpad=10)
    axes.set_ylabel('%s'%(bootstrap_measure.split('_')[0]))
    axes.set_xticklabels(['BLOSUM50','C_S_H', 'OH', 'OHF', 'OHM'])
    
    out_dir=f"../data/{allele}_out"
    out_n=f"boxplot_{allele}_{bootstrap_measure.split('_')[0]}"
    fig.savefig(os.path.join(out_dir,out_n),dpi=200)   
    
    
        
        
def barplot_oneallele(df, allele, train_size, perf_measure, errorbar):
    
    # set width of bars
    barWidth = 0.25
    
    #for allele, d_ in df.groupby('Allele'):
    d_ = df.loc[df["Allele"] == allele].loc[df["Train_size"]==1976]
        
    # initialize plot
    fig, axes = plt.subplots()

    # initalize variables for the positions of the bars dynamically depending how many different
    # encoding schemes that we are using
    # the size needed is numb of encodings * barWidth + space between
    #r = [0+r*(barWidth*len(d_['Encoding'].unique())+2*barWidth) for r in np.arange(len(d_["Train_size"].unique()))]
    r = [0+r*(barWidth*len(d_['Encoding'].unique())+1*barWidth) for r in [0]]
    
    for encod, d in d_.groupby('Encoding'):
    #d = d_.loc[d_["Train_size"]==int(train_size)]
        
        # Set position of bar on X axis
        # we loop over the different encodings and we need to update 
        # the position of the different bar posithin with + a barWidth 
        r = [x + barWidth for x in r]
    
        # plot the bars for the different encodings and test sizes
        axes.bar(r, d[perf_measure], width=barWidth, yerr=d[errorbar], label=encod, capsize=3, alpha=0.8, ecolor='darkslategray')
    
            
    # make the plot look pretty
    axes.set_title("Performance of different encodings for allele: %s"%allele, pad=10)
    axes.set_xlabel('Training set size')
    axes.set_ylabel('%s'%perf_measure)
    if perf_measure == 'AUC':
        axes.set_ylim([0.60,1])
    if perf_measure == 'MCC':
        axes.set_ylim([-0.05,0.9])
    # we are adjusting the x tick location from the last r (bar is in the middle) so we need to add
    # half a barWidth and then subtract the barWidth * numb of encoding so we move the tick to the middle
    axes.set_xticks([p+barWidth/2-(barWidth*len(d_['Encoding'].unique())/2) for p in r])
    axes.set_xticklabels(d_["Train_size"].unique())
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), fancybox=True, ncol=3)
    fig.tight_layout()
    
    out_dir=f"../data/{allele}_out"
    out_n=f"bar_{allele}_only_{perf_measure}-{'-'.join(d_['Encoding'].unique())}"
    fig.savefig(os.path.join(out_dir,out_n),dpi=200)
    #plt.show()