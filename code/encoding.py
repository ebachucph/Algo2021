import numpy as np
import pandas as pd

MAX_PEP_SEQ_LEN = 9

def load_scheme(filename):
    """
    Read in encoding scheme values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def encode_peptides(Xin, scheme_file):
    """
    Encode AA seq of peptides using BLOSUM50.
    Returns a tensor of encoded peptides of shape (batch_size, MAX_PEP_SEQ_LEN, n_features)
    """
    encoding_scheme = load_scheme(scheme_file)
    
    batch_size = len(Xin)
    n_features = len(encoding_scheme)
    
    Xout = np.zeros((batch_size, MAX_PEP_SEQ_LEN, n_features), dtype=float) # Use float to accomodate the different schemes
    
    for peptide_index, row in Xin.iterrows():
        for aa_index in range(len(row.peptide)):
            Xout[peptide_index, aa_index] = encoding_scheme[ row.peptide[aa_index] ].values
            
    return Xout, Xin.target.values
