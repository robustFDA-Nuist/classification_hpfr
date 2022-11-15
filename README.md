The codes and a demo example of Melbourne Pedestrian Data for the manuscript  "Joint Classiffication and Prediction of Random Curves
Using Heavy-tailed Process Functional Regression".

The following files are stored in .../classification_hpfr/

demo.m : the matlab code for a demo example of  Melbourne Pedestrian Data. 
traindata.mat : training data file. In which: (assume that the number of  training curves is n and the number of observations of curves is m)
                         traindata: K x 1 cells structure. The k-th cell is a matrix in which observations of the variable y in k-th class.
	
testdata.mat : testing data file. In which: 
                         Struct, in which 'testdata' is the observations of the new subjects y for classification and prediction and 'label' is the label of the new subjects y.



fdaM: J.O.Ramsay's FDA codes with minor changes

classificaiton_hpfr: codes for Joint classiffication and prediction of random curves
using heavy-tailed process functional regression
