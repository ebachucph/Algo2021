# Project overview
## Project parts
1. Parser
  * Pandas
  * Data files are text files (link in paper Peters et. al. 2006)
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
1. GitHub repo
2. Markdown file for report
3.	Make parser script
  * To parse peptide data
    * Replicate method in exercises in Pandas
  * To parse encoding schemes
    * Replicate method in exercises in Pandas
4. Finding data for encoding schemes / make PSSM’s
  * Make PSSMs of Charge, Size, etc.
5. Constructing NN
6. Make evaluation script
 * Comparing different alelles
 * Statistics
 * Encoding scheme complexity?
 * Plot number of binders versus encoding schemes
 * Performance on y and size of data set on x - comparing the encoding schemes 

## Discussion notes
-	Hydrophobicity and charge relative change depending on the peptide composition
-	Homology reduction (also in regards to size/charge)
