# Project overview
## Project parts
1. Parser
  * Pandas
  * Data files are text files (link in paper Peters et. al. 2006)
  * Peptide data will be retrieved from the SMM exercise to save time...
2. Encoding schemes
  * One-hot
  * Modified one-hot
  * BLOSUM
  * Charge
  * Size
  * Hydrophobicity
  * ?
3. Network
  * Individual or combined with different encodings?
  * Ensemble of networks?
4. Evaluation

## Structure
- 1 main script to call to all other scripts:
  * Parsing
  * Encoding
  * Training
  * Evaluating
    *  Prediction of experimental binding affinities
    *  Classify binders/non-binders
    *  AUC
- Helpers
  * Define functions above here
  * If too big, split up.

## To-Do
1. GitHub repo (Done)
2. Markdown file for report
3.	Make parser script
  * To parse peptide data (Alex)
    * Replicate method in exercises in Pandas
  * To parse encoding schemes (Emil)
    * Replicate method in exercises in Pandas
4. Finding data for encoding schemes / make PSSM’s (Emil)
  * Make PSSMs of Charge, Size, etc.
5. Constructing NN (Lasse)
6. Make evaluation script (Alex)
 * Comparing different alelles
 * Statistics
 * Encoding scheme complexity?
 * Plot number of binders versus encoding schemes
 * Performance on y and size of data set on x - comparing the encoding schemes 

## Discussion notes
*	Hydrophobicity and charge relative change depending on the peptide composition
*	Homology reduction (also in regards to size/charge)

## Introduction notes
* Peptide MHC binding and importance (Emil)
  * What's going on
  * Factors that are important
  * T-cell - MHC interaction (short)
* Introduction to the different encoding schemes
  * SPARSE/ONE_HOT
  * Charge (Emil)
  * Size (Emil)
  * Hydrophobicity (Emil)
  * BLOSUM (Emil)
* Neural Networks (Lasse)
* Aim
  
## Results notes
* Overview comparing different encoding schemes in line plot
* 
