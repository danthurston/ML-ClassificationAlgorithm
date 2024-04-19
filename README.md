# Machine Learning Classification Algorithm

## Introduction
For this assignment the task was to develop a machine learning classification algorithm to find the percentage of spam in a dataset of thousands of emails. Once this had been achieved, a n-fold cross validation will test it's effectiveness and statistics regarding accuracy are gathered.

## Preparing the Data
The first task is to prepare the data for processing by the classification algorithm. This starts with importing the CSV containing the dataset. To handle the data, the Pandas library is imported and utilized for CSV file handling. 

### Splitting
The data is then split into training and testing sets using ‘scikits’ built-in ‘train_test_split’ function. This function is sent the target data (column) and the remainder of the data as two separate sets. This was performed using the ‘iloc[ ]’ function to select certain pieces of data from the set. These sets are then passed to the train_test_split function to create training and testing sets. If the model was trained on all the data then it would have near one hundred percent accuracy, however, any new data added would likey reduce the accuracy. This is known as over-fitting. Similarly, a model can suffer from under-fitting if the split percentages are too far in the other direction. It is ideally an 80/20 percent split, with ‘train’ taking the majority (McCaffrey, 2013).

### Scaling
As data can come in different forms of magnitude and ranges, there is a need to scale down the data so it’s more manageable, this is known as standardization or normalization. Ideally the data will be between 0 and 1. Scaling is important as significant data differences can cause larger data to overpower other data. For this program, the ‘StandardScaler()’ function was used to implement scaling. This function normalizes data so that the mean dataset value will be 0 and standard deviation will be 1. The data is centered and scaled separately on each feature by computing relevant statistics on the training set data. Mean and standard deviation are stored to be used with other data using the transform method (Sklearn, 2019). The scaled data is now ready to be passed to the classification algorithm.

## Classification Algorithm
The Multi-Layer Perceptron (MLP) neural network is used due to its effectiveness and accuracy when handling vast amounts of data, though computing time can become an issue.
To develop this algorithm for the dataset and ensure accuracy, the correct parameters need to be chosen. The MLP classifier is comprised of a multitude of parameters that impact the output accuracy in some way (appendix 1). With so many parameters involved it can be a complex and lengthy process to find the perfect value for each parameter. As such, a technique to find the ideal parameters for the dataset can be used. This is known as hyperparameter optimization.

## Hyperparameter Optimization
Due to the complexity of the chosen algorithm, many parameters could influence the accuracy of the model. Hyperparameter optimization (HPO) allows all the parameters to have various options tested, running through all possible permutations and finally displaying the optimum classifier. The HPO function chosen for this task was GridSearchCV. One of the main benefits of this function is that it also performs cross-validation (CV).

HPO can be a lengthy task with so many variations of the classifier to run. To aid in this, certain parameters can be removed if they are not applicable. The more parameters that can be chosen prior to HPO the better as it massively decreases the computing time when running the function. Research into which parameters to choose based on the dataset and circumstances can be performed to aid in this. For example, the solver ‘adam’ was chosen due to its known effectiveness with datasets containing thousands of entries. Making this decision ruled out many of the other parameters as some only apply to specific solvers. Time complexity will grow exponentially with every added parameter, thus they should be reduced as soon as possible. The process of breaking down the classifier into its components to aid in designing the perfect model is performed in stages as follows. 

### Breakdown
First is the full classifier with all its parameters at its default state (Appendix 1):

![1-MLPClassifier](https://user-images.githubusercontent.com/54746562/141026247-a4d2d5b9-00a5-4794-bfc9-09a8831e590a.png)

#### Stage 1:
List all parameters in an easier to manage format.

![2](https://user-images.githubusercontent.com/54746562/141026320-8bf033e2-5424-47a5-ab67-c6d9cf54dc68.png)

#### Stage 2:
The solver ‘adam’ being chosen excludes certain parameters that only apply to other solvers.

![3](https://user-images.githubusercontent.com/54746562/141026404-080333bb-cc9f-4ddf-be72-d3bab7207ec1.png)

#### Stage 3:
All parameters that are not applicable, or the effects of which are not desired can be removed. Other parameters are useful for testing purposes but not required in practice.

![4](https://user-images.githubusercontent.com/54746562/141026478-7eccd1c1-7580-48a3-877b-a805cccc5134.png)

#### Stage 4:
This is the result of the breakdown. A list of all the parameters that need to be passed to the HPO to find their optimum values. This reduction in parameters results in a much more manageable list and hyperparameter optimization will be much less complex to implement.

![5](https://user-images.githubusercontent.com/54746562/141026519-ccb4544c-bd7a-4ace-80dd-277c0e6f8cee.png)

### HPO Function: GridSearchCV()
Once the breakdown process has been completed the parameter space can be defined. This is a list of all parameters and their respective values for cross referencing to find the ideal classifier. For most tests the parameters had their testing variables set slightly either side of the default, or previous result, to be able to decipher whether they should be raised or lowered. It is interesting to note that this process can utilize multiple cores and so the completion time is dependent on the hardware available.

![6](https://user-images.githubusercontent.com/54746562/141026550-4e2c81cb-b4b1-439e-b68c-488763d0987d.jpg)

![7](https://user-images.githubusercontent.com/54746562/141026570-ab98d5e9-44a2-4b80-a7d2-6f3f6cceac7c.png)

The result of this process are the parameters GridSearchCV defines as ideal for the dataset from the variables input. Further tests are run with various parameters and their respective values to find the perfect classifier for the data. It should be noted that ideally all the parameters would be tested at once with various values to try all possible permutations, however, this is impractical due to the exponential increase in complexity with each additional input.

Using this method of smaller batches of parameters, it's vital to notice trends and pick up on consistent outcomes. For example, every test that included the ‘activation’ parameter returned ‘relu’ as the optimum value, so this could reliably be removed from the list and another parameter added. This process is continued until all parameters can be fixed to their ideal values (see appendix 3 for more test data). 

After all the parameters are selected, the param_grid variable of GridSearchCV is set to null so that only cross-validation is performed on the completed MLPClassifier for the final output.

## Cross-Fold Validation Technique
The general idea behind n-fold cross validation is to split the entire dataset into equal sets, with ‘n’ being the number of groups to split the data into. Each group is used as the test set one-by-one whilst the remainder of the data is used as the training set. The score of the test is retained, the model discarded, and the function moves onto the next test set. Once all tests have been run, all the scores are cross referenced to give the model an accuracy score. 

Once cross-validation has begun, the data groups are fixed in their respective sets for so that each group has the chance to be the test set once and the training set n-1 times. The main choice to make with cross validation is the number of folds (n) to use to validate the model. Generally this number is 5 or 10 as these values have been shown to result in neither high bias nor variance (Brownlee, 2018). For this program, 5-fold was chosen over 10 as the latter would be excessive and mainly result in an increase in processing time.

## Accuracy Score and Evaluation
If a high score is achieved on one set of test data, this does not denote complete success as another data set could be completely off. The aim is to achieve a steadily high and consistent accuracy rate through numerous tests.

There are three main metrics used as the final output for this program.
1. __accuracy_score__, which is the fraction of samples that were predicted correctly.
2. __F1_score__, which is a harmonic mean between precision and recall. It's formula is as follows: F1 = (2 * (precision * recall)) / (precision + recall).
3. __roc_auc_score__ is essentially the ‘area under a ROC curve’. ROC stands for Receiver Operating Characteristics. A ROC curve is formed by plotting the true positive rate (TPR) (probability of detection) against the false positive rate (FPR) (probability of false alarm). The area under such curves are often used as a measure of performance, especially for classification problems.

Following these three statistic outputs is a **confusion matrix**, also known as an error matrix. This is a table that’s commonly used in machine learning to display the error rate of the program. As can be seen in the prediction table (which utilizes the example output below), these tables cross-reference predicted and actual yes and no outputs, which is extended to show totals.

![8](https://user-images.githubusercontent.com/54746562/141026616-11671ed0-3f11-4ec0-a2f9-e1b5935c4919.png)
![9](https://user-images.githubusercontent.com/54746562/141026645-ad06e13b-ddc7-4158-8485-c61e25143e8b.png)

The final section of the output is a **classification report** which displays various data to aid in the evaluation of the model. Precision is the fraction of the results that are relevant. Recall is the fraction of total relevant correctly classified results by the program. To aid in the understanding of precision and recall, a few tables and diagrams are provided below. The support column displays the total values, as seen in the confusion matrix (Saxena, 2019).

![10](https://user-images.githubusercontent.com/54746562/141026686-8ac70161-6100-44ca-9f3a-dad9d35efb55.png)
![11](https://user-images.githubusercontent.com/54746562/141026726-350e05c3-8663-4353-8702-59f18e483b02.png)

## Conclusion
The final program is incredibly concise, being only 100~ lines of code, whereas if this was coded in Java instead of Python it would undoubtedly be hundreds of lines of code and inherently less efficient. With the test computer only able to run HPO on 7 or so parameters at a time, they had to be broken down into small batches. Despite the breakdown meaning a huge increase in time spent running tests, it did mean that a far deeper understanding of all the classifiers’ parameters could be achieved. For example, it was discovered which parameters had strong effects or links to one another and so needed to be tested together, such as ‘max_iter’ and ‘tol’. 

Attempting to find ideal parameters using GridSearchCV() took an substantial amount of processing power, sometimes having to leave my laptop running for days. For example, the HPO function with the parameters shown below was left to run for __91 hours__ before deciding to reduce the number of parameters and breakdown the process to make the function faster.

![12](https://user-images.githubusercontent.com/54746562/141026754-15333c3b-cc61-4120-b8fc-7f0d5340e6bf.png)

![AP3-searchtest](https://user-images.githubusercontent.com/54746562/141694757-5181bd06-d370-4503-9542-5f80fcbcb783.png)

## References:
Brownlee, James - Machine Learning Mastery. 2019. A Gentle Introduction to k-fold Cross-Validation. [ONLINE] Available at: [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/). [Accessed 15 May 2019].

McCaffrey, James - Visual Studio Magazine. 2013. Understanding and Using K-Fold Cross-Validation for Neural Networks. [ONLINE] Available at: [https://visualstudiomagazine.com/articles/2013/10/01/understanding-and-using-kfold.aspx](https://visualstudiomagazine.com/articles/2013/10/01/understanding-and-using-kfold.aspx). [Accessed 14 May 2019].

Shruti Saxena – 2019. Precision vs Recall – Towards Data Science. [ONLINE] Available at: https://towardsdatascience.com/precision-vs-recall-386cf9f89488. [Accessed 21 May 2019]
Scikit-learn - 2019. sklearn.neural_network.MLPClassifier — scikit-learn 0.21.1 documentation. [ONLINE] Available at: [https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). [Accessed 15 May 2019].

Scikit-learn - 2019. sklearn.model_selection.GridSearchCV— scikit-learn 0.21.1 documentation. [ONLINE] Available at: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). [Accessed 15 May 2019].

Scikit-learn - 2019. sklearn.preprocessing.StandardScaler— scikit-learn 0.21.1 documentation. [ONLINE] Available at: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). [Accessed 15 May 2019].

## Appendix

### Appendix 1 - Parameter Breakdown - MLP Classifier 
Even after reorganising the full classifier, it doesn’t always help with defining each parameters purpose. What follows is a breakdown of all the parameters of the ‘MLPClassifier’. Creating this aided in research prior to creating the classifier and as an efficient reference.

![AP1-AllParameters](https://user-images.githubusercontent.com/54746562/141694468-0790ac8d-32af-4763-9883-93dcac09f3df.png)

### Appendix 2 - Parameter Breakdown - GridSearchCV
Breaking down the GridSearchCV function enables a more complete understanding of the function at hand and allows an understanding of all the abilities of the function. GridSearchCV is not as complex as the MLPClassifier however, it still benefitted the project by being broken down and analysed.

![AP2-GridsearchCV](https://user-images.githubusercontent.com/54746562/141694604-8f830fc7-2f50-4262-a5cb-cf051e1b3510.png)

### Appendix 3 - Further Testing
The following is a small example of tests ran to find optimum parameters using hyperparameter optimization with GridSearchCV. 

#### Example 1
![AP4-further](https://user-images.githubusercontent.com/54746562/141695231-dcf048ad-3334-4107-b65b-2f497b809eda.png)

#### Example 2
![AP5-further2](https://user-images.githubusercontent.com/54746562/141695246-d8c3d565-ac4d-4d86-b9ee-7c589470041d.png)

#### Example 3
![AP5-further3](https://user-images.githubusercontent.com/54746562/141695265-767b640f-7cb4-4a82-b05d-28f0bda34058.png)

#### Example 4
![AP5-further4](https://user-images.githubusercontent.com/54746562/141695285-1e3f3702-a285-4e78-b355-f51ce98f5efb.png)



