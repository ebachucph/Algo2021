import numpy as np
import pandas as pd
from os.path import dirname
from os import listdir

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
	
	batch_size = len(Xin)
	n_features = 21 # All encoding schemes have the same structure
	Xout = np.zeros((batch_size, MAX_PEP_SEQ_LEN, n_features), dtype=float) # Use float to accomodate the different schemes
	
	# Check if CHARGE is in scheme_file, if true load all CHARGE schemes and encode in special way
	if "CHARGE" in scheme_file:
		scheme_dir = dirname(scheme_file)
		scheme_files = listdir(scheme_dir)
		
		encoding_schemes = []
		
		for f in scheme_files:
			if "CHARGE" in f:
				encoding_schemes.append(load_scheme(scheme_dir + "/" + f))
		
		for peptide_index, row in Xin.iterrows():
			for aa_index in range(len(row.peptide)):
				
				# Append  values for charge of side chain
				Xout[peptide_index, aa_index] = encoding_schemes[0][row.peptide[aa_index]].values
				# If this amino acid is the N-terminal add the charge of the N-terminal amino-group
				
				if aa_index == 0:
					Xout[peptide_index, aa_index] += encoding_schemes[2][row.peptide[aa_index]].values
				
				# If this amino acid is the C-terminal add the charge  of the C-terminal carboxyl-group
				if aa_index == len(row.peptide) - 1:
					Xout[peptide_index, aa_index] += encoding_schemes[1][row.peptide[aa_index]].values
			
	# Load other schemes normally		
	else:
		encoding_scheme = load_scheme(scheme_file)
		
		for peptide_index, row in Xin.iterrows():
			for aa_index in range(len(row.peptide)):
				Xout[peptide_index, aa_index] = encoding_scheme[ row.peptide[aa_index] ].values
		        
	return Xout, Xin.target.values
