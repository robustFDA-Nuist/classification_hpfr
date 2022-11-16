"Joint Classification and Prediction of Random Curves Using Heavy-tailed Process Functional Regression"

We propose a heavy-tailed process functional regression to jointly perform classification and prediction of time-varying functional data. We use two independent scale mixtures of Gaussian Processes to respectively model random effects and random errors, yielding robust inferences against both magnitude and shape outliers. We classify random curves by posterior predictive probabilities of class labels and offer a weighted prediction of future curve trends. A Bayesian estimation procedure is implemented through an MCMC sampling algorithm. The codes and a demo example of Melbourne pedestrian data for the manuscript  "Joint Classiffication and Prediction of Random Curves Using Heavy-tailed Process Functional Regression". 

classificaiton_hpfr: codes for Joint classiffication and prediction of random curves using heavy-tailed process functional regression. To run this code, you need to load
the 'fdaM' file first. In which: 'classification_hpfrtrain.m' is the training code for parameter estimation and 'classification_hpfrpred.m' is the testing code for classification and prediction. 

demo.m :       the matlab code for a demo example of Melbourne pedestrian data. In which, we perform classification and prediction and obtain the classification
accuracies and predictive RMSEs.

traindata.mat : training data file. In which: traindata: K x 1 cells structure. For example, the cells correspond to the training data of Bourke Street Mall (North),
Chinatown-Swanston St (North), Southbank and Southern Cross Station, respectively.
	
testdata.mat : testing data file. In which: Struct structure, in which 'testdata' is the observations of the new subjects for classification and prediction and 'label'
are labels of the new subjects.



fdaM: J.O.Ramsay's FDA codes with minor changes, which can be availabled at https://www.psych.mcgill.ca/misc/fda/downloads/FDAfuns/Matlab/.
