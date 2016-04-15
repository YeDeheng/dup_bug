Predicting Priority Level of Bug Reports

Sample Datasets consists of 2,000 bug reports downloaded from Eclipse Bugzilla within a time period, from 1st January 2008 to 31st December 2008.

The bug report have the following format:
ID:
Summary:
Description:
Product:
Component:
Version:
Priority:
Severity:

There are five priority levels: P1, P2, P3, P4, and P5, with P1 having the highest importance, to P5 with the lowest importance.

Bug reports are split into training set and testing set with 1,000 bug reports each.
The priority field of bug reports in the testing set are removed.

Train_Multiple_Files.py, and stopwords.py should be put in the same location.

Scikit-learn is used in Train_Multiple_Files.py to perform machine learning task.
All inputs to the machine learning task in scikit-learn needs to be in the following format (2D array, with (n_samples, n_features)):
[[  0.   0.   5. ...,   0.   0.   0.]
 [  0.   0.   0. ...,  10.   0.   0.]
 [  0.   0.   0. ...,  16.   9.   0.]
 ...,
 [  0.   0.   1. ...,   6.   0.   0.]
 [  0.   0.   2. ...,  12.   0.   0.]
 [  0.   0.  10. ...,  12.   1.   0.]]


Eclipse Bugzilla: https://bugs.eclipse.org/bugs/
Scikit-learn: http://scikit-learn.org/stable/
