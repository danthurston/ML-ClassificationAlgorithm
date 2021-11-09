from sklearn.preprocessing import StandardScaler		# for scaling down values non-binary
from sklearn.model_selection import train_test_split	# for training data
from sklearn.neural_network import MLPClassifier		# import MLP
from sklearn.model_selection import GridSearchCV		# HPO grid search method
from sklearn.model_selection import KFold 		     	# for n-fold cross validation
from sklearn import metrics			 					# for evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix # for confusion matrix
from numpy import array
import pandas as pd								    	# for reading CSV
#import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
import datetime

# Print start time
currentDT = datetime.datetime.now()
print("\nStart Time: " + (currentDT.strftime("%Y-%m-%d %H:%M:%S")))

# disable data conversion warning (for scaler)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

border = "--------------------------------------------------------------------\n"

def underline( str ):			                    # function to underline words with dashes
	line = "-" * len(str)							# number of dashes = length of string
	return line

# Import data #
dataset = pd.read_csv("dataset.csv", header=None)   # import dataset from CSV
# Print title and head (top 5 rows of data) enclosed in borders 
title = "Head of Dataset:"
print("\n" + border + title + "\n" + underline(title) + "\n", dataset.head(), "\n" + border)

# define target and data sets
target = dataset.iloc[:,-1]                         # everything minus the target column << CHECK
data   = dataset.iloc[:,0:57]                       # target column

# create training and testing variables
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

title = "Training and Testing Sets:"
print(border + title + "\n" + underline(title))     # print title with underline
print(X_train.shape,y_train.shape)					# print train sets
print(X_test.shape, y_test.shape, "\n" + border)    # print test sets and border

# Scale data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True) 
scaler.fit(X_train)									# fit scaler with train set  
X_train = scaler.transform(X_train)					# update X_train to a scaled set
X_test  = scaler.transform(X_test)					# update X_test to a scaled set

# MLP Classifier #
mlp = MLPClassifier(hidden_layer_sizes=(135,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', beta_1=0.8, beta_2=0.999, learning_rate='adaptive', max_iter=11000, shuffle=True, random_state=None, verbose=False, warm_start=False, early_stopping=False, nesterovs_momentum=False, n_iter_no_change=11, learning_rate_init=0.001, tol=0.001, epsilon=1e-08)

#create a trained model from the classifier
trainedModel = mlp.fit(X_train,y_train)
title = "Trained Model Score:"
print(border + title +  "\n" + underline(title))	    # print title with underline
print(trainedModel.score(X_test,y_test))			# print the accuracy score of the trained model
print(border)

# Run hyper-parameter optimization to find best parameters #
	# parameters to compare:
parameter_space={}									# here is where  parameters for testing were listed

	# execution of HPO test, with N-Fold Cross-Validation:
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, iid='warn', refit=True, verbose=0, error_score='raise-depreciating', return_train_score='warn')

result = clf.fit(X_train, y_train)

    # print HPO parameters:
##title = "Best parameters found:"
#print(border + title + "\n" + underline(title) + "\n", clf.best_params_)
#print(border + "\nBest Score:\n", clf.best_score_)
#print("\nBest Estimator:\n", clf.best_estimator_, "\n")
#print(border)
##

pred = clf.predict(X_test)

# Classification algorithm evaluation metrics #
acc = metrics.accuracy_score(y_test, pred)				
f1  = metrics.f1_score(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)

print(border + "Metrics Accuracy Score: ", acc)			
print("Metrics f1 Score :      ", f1)
print("Metrics Auc Score:      ", auc, "\n" + border)

# Confusion matrix:
title = "Confusion Matrix:"
print(border + title +  "\n" + underline(title))
print((confusion_matrix(y_test,pred)), "\n" + border)

# Classification Report
title = "Classification Report:"
print(border + title + "\n" + underline(title))  
print((classification_report(y_test,pred)), "\n" + border)

# Print finish time
currentDT = datetime.datetime.now()
print("Finish Time: " + (currentDT.strftime("%Y-%m-%d %H:%M:%S")))