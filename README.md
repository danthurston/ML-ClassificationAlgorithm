# Machine Learning Classification Algorithm

## Introduction
For this assignment the task was to develop, define and utilize a machine learning classification algorithm. The algorithm was ultimately to be used to find the percentage of spam in a data set of thousands of emails. Once this had been achieved, the following task was to apply n-fold cross validation to the algorithm to test its effectiveness across the whole dataset. Lastly, statistics regarding the accuracy of the algorithm are output for a final evaluation.

## Preparing the Data
The first task was to prepare the data for processing by the classification algorithm. This starts with importing the CSV containing the dataset. To handle the data, the Pandas library is imported and utilized as it is known for being spectacular at handling CSV files. 

## Splitting
From here, the data is split into training and testing sets of data using ‘scikits’ built-in ‘train_test_split’ function. This function is sent the target data (column) and the remainder of the data as two separate sets. This was performed using the ‘iloc[ ]’ function to select certain pieces of data from the set. These two sets are then passed to the train_test_split function to create training and testing sets of data. The data is split because if the model was trained on all the data then it would have near one hundred percent accuracy. However, if new data is added, then it’s accuracy would likely be very poor. This is known as over-fitting. Similarly, a model can suffer from under-fitting if the split percentages are too far in the other direction. It is ideally an 80/20 percent split with ‘train’ taking the majority (McCaffrey, 2013).

## Scaling
As data can come in forms of various magnitude and ranges, there is a need to scale down the data so it’s more manageable, this is also known as standardization or normalization. Ideally the data will all be between 0 and 1. Scaling is important as when data has massive differences, the larger data can overpower the other data in the set. For this program, the ‘StandardScaler()’ function was used to implement scaling. This function normalizes the data in such a way that the mean value of the dataset will be 0 and standard deviation will be 1. The data is centered and scaled separately on each feature by computing relevant statistics on the training set data. Mean and standard deviation are stored to be used on later data using the transform method (Sklearn, 2019). Even variables that were thousands away from others are now scaled to a proportionate and manageable level. The data is now ready to be passed to the chosen classification algorithm.

## The Classification Algorithm
The chosen classification algorithm for this task was the Multi-Layer Perceptron (MLP) Neural Network. It was chosen due to its powerful effectiveness and accuracy. Even though computing time can become an issue, an advanced algorithm such as this was ideal due to its ability to handle vast amounts of data.
To develop this algorithm for the dataset the correct parameters had to be chosen to ensure accuracy. The MLP classifier is comprised of a multitude of different parameters that each affect the accuracy of the output by one means or another (see appendix 1). The key to creating the perfect classifier is choosing the perfect parameters. As there are so many parameters involved with neural networks, it can be a complex and lengthy process to find the perfect value for each parameter. As such, a technique to find the ideal parameters for the dataset can be used which is known as hyperparameter optimization.

## Hyper-Parameter-Optimization
Due to the complexity of the chosen algorithm, there are many parameters that can influence the accuracy of the model. With so many variables and every variant of the MLP they could produce, it can be difficult to choose the right parameters. To aid in finding the optimum inputs, hyper-parameter-optimization (HPO) allows all the parameters to have various options tested, running through all possible permutations and finally displaying the optimum classifier. The HPO function chosen for this task was GridSearchCV. One of the main benefits of this function is that it also performs cross-validation (as denoted by the ‘CV’).

Especially in neural networks with huge datasets, HPO can be a very long task to complete as so many different variations of the classifier need to be run. To aid in this, certain parameters can be removed by default because they are non-applicable to the dataset or they do not apply to some fixed parameter that has been chosen prior to HPO. The more parameters that can be chosen prior to HPO the better as it massively decreases the computing time when running the function. Research into which parameters to choose based on the dataset and circumstances can be performed to reduce complexity.
An example of the above is that the solver ‘adam’ was chosen due to its known effectiveness with datasets containing thousands of pieces of data. Making this decision ruled out many of the other parameters as some only apply to specific solvers. Time complexity will grow exponentially with every added parameter (and respective variables), thus the need to reduce the number of them as soon as possible. The process of breaking down the classifier into its components to aid in designing the perfect model is done in several stages and breaks down as follows. 

## Breakdown
First, we begin with the full classifier with all its parameters at its default state (Appendix 1):
![1-MLPClassifier](https://user-images.githubusercontent.com/54746562/141026247-a4d2d5b9-00a5-4794-bfc9-09a8831e590a.png)

### Stage 1:
List all parameters for an easier to work with format.
![2](https://user-images.githubusercontent.com/54746562/141026320-8bf033e2-5424-47a5-ab67-c6d9cf54dc68.png)

### Stage 2:
The solver ‘adam’ being chosen excludes certain parameters that only apply to other solvers.
![3](https://user-images.githubusercontent.com/54746562/141026404-080333bb-cc9f-4ddf-be72-d3bab7207ec1.png)

### Stage 3:
All the parameters that are not applicable, or the effects of which are not desired can now be removed. Other parameters are useful for testing purposes but not required in practice.
![4](https://user-images.githubusercontent.com/54746562/141026478-7eccd1c1-7580-48a3-877b-a805cccc5134.png)

### Stage 4:
This is the result of the breakdown. A list of all the parameters that need to be passed to the HPO to find their optimum values. This reduction in parameters results in a much more manageable list than initially seen. From here, the hyper-parameter optimization function will be much less complex to implement.
![5](https://user-images.githubusercontent.com/54746562/141026519-ccb4544c-bd7a-4ace-80dd-277c0e6f8cee.png)

## HPO Function – GridSearchCV()
The chosen function to execute HPO is GridSearchCV(). Once the breakdown process has been completed the ‘parameter_space’ can be defined. This is essentially the list of all the parameters and their various inputs to test to find the perfect classifier. For most tests, the chosen parameters had their testing variables set slightly either side of the default (or the previous result) to be able to decipher whether they are required to be set higher or lower.

For example, this test was comprised of the following parameters and their chosen variables. 
To show the pure computational power required to complete this task; a laptop with 8GB RAM, Dual-Core AMD-A6 processor offline and running nothing but the referenced program had to be run overnight. Before being left to its devices overnight, it had already been running for 9 hours. It is interesting to note that this function can utilize multiple cores and so if this program was run on a quad-core computer the completion time would be halved.

![6](https://user-images.githubusercontent.com/54746562/141026550-4e2c81cb-b4b1-439e-b68c-488763d0987d.jpg)
![7](https://user-images.githubusercontent.com/54746562/141026570-ab98d5e9-44a2-4b80-a7d2-6f3f6cceac7c.png)

The final output of this process is what the function defined as the perfect parameters, along with its ideal complete classifier. From here, further tests are run with various parameters and their respective values to find the perfect classifier for the data. It should be noted that ideally all the parameters would be tested at once with various values to try all possible permutations, however, without a more powerful processor, this could take days to weeks due to the exponential increase in time with parameter value additions (see reflection).

Using this method of smaller batches of parameters, it was vital to notice trends and pick up on constant outcomes. For example, on every test run that included the ‘activation’ parameter, ‘relu’ was returned as the optimum value and so this parameter could reliably be removed from the list and another parameter added in its place. This process is continued until all the parameters can be locked down to their ideal values (see appendix 4 for more testing inputs and outputs). 
After all the parameters are selected, the ‘param_grid’ variable of GridSearchCV is set to null so that it just performs cross-validation on the completed MLPClassifier for the final output. The program is now ready for final cross-fold validation and evaluation statistics.

## Cross-Fold Validation Technique
Once theAs previously mentioned, the GridSearchCV function used to find the optimum parameters also implements cross-fold validation. The general idea behind n-fold cross validation is to split the entire dataset into equal sets. The ‘n’ is the number of groups that the data will be split into. Each of these sets are then used as the test set one-by-one whilst the remainder of the data is used as the training set. The score of this test is retained and the model discarded, and the function moves onto the next test set. Once all tests have been run, all the scores are cross references to give an accuracy score of the model. 

It’s important to note that all once the process of cross-validation has begun, the sets of data are fixed into their respective sets for the duration so that each grouping has the chance to be the test set once and the training set n-1 times. The main choice to make with cross validation is the number of folds (n) to use to validate your model. Generally, this number is 5 or 10 as these values have been shown to result in neither high bias nor variance. (Brownlee, 2018). For this program, 5-fold was chosen over 10 as the latter would be excessive and mainly result in an increase in processing time.

## Accuracy Score and Evaluation
As the final output are a series of tools to aid in the evaluation of an algorithm. It’s important to note that just because a high score is achieved on one set of test data that this does not denote complete success as another set of data could be completely off. The aim is to achieve a steadily high and consistent accuracy rate through numerous tests.

There are three main metrics used as the final output for this program, accuracy_score, which is simply the fraction of samples that were predicted correctly. F1_score is a harmonic mean between precision and recall, its formula is as follows: F1 = (2 * (precision * recall)) / (precision + recall). Finally, roc_auc_score is essentially the ‘area under a ROC curve’. ROC stands for Receiver Operating Characteristics. A ROC curve is formed by plotting the true positive rate (TPR) (essentially probability of detection) against the false positive rate (FPR) (essentially probability of false alarm). The area under such curves are often used as a measure of performance, especially for classification problems.

Following these three statistic outputs is a confusion matrix, also known as an error matrix. This is a table that’s commonly used in machine learning to display the error rate of the program. As can be seen in the table below (which utilizes the example output to the side), these tables cross-reference predicted and actual yes and no outputs, which is extended to show totals.

![8](https://user-images.githubusercontent.com/54746562/141026616-11671ed0-3f11-4ec0-a2f9-e1b5935c4919.png)
![9](https://user-images.githubusercontent.com/54746562/141026645-ad06e13b-ddc7-4158-8485-c61e25143e8b.png)

The last output is a classification report which displays various averages and info to aid in the evaluation of the mode. Precision is the fraction of the results that are relevant. Secondly, Recall is the fraction of total relevant correctly classified results by the program. To aid in the understanding of precision and recall, a few tables and diagrams are provided below. As for the final column, it can be seen using the confusion matrix example table, the ‘support’ column displays the total values (Saxena, 2019).

![10](https://user-images.githubusercontent.com/54746562/141026686-8ac70161-6100-44ca-9f3a-dad9d35efb55.png)
![11](https://user-images.githubusercontent.com/54746562/141026726-350e05c3-8663-4353-8702-59f18e483b02.png)

## Conclusion
The final program is incredibly concise, it’s only 100 or so lines of code, whereas if this was coded in Java instead of Python it would undoubtedly be hundreds of lines of code and inherently far less efficient. It should also be noted that due to the computer running the program being unable to run a complete HPO search, it cannot be said that the optimum MLP classifier was definitely found. With the computer only able to run HPO on 7 or so parameters at a time, they had to be broken down into small batches. Despite the breakdown meaning a huge increase in time spent running tests, it did mean that a far deeper understanding of all the classifiers’ parameters was achieved. This meant that it was discovered which parameters had strong effects or links to one another and thus needed to be tested together, such as ‘max_iter’ and ‘tol’ which both interrelate. 

## Reflection
The decision to choose the neural network classifier MLP was made because there has been a huge surge in popularity in the realm of neural networks (and deep learning) in recent years and so practice in this field was something very desirable. When testing and attempting to find the correct parameters using the ‘GridSearchCV()’ method, this took an extremely large amount of computing power, sometimes having to leave my laptop running the program for days.
![12](https://user-images.githubusercontent.com/54746562/141026754-15333c3b-cc61-4120-b8fc-7f0d5340e6bf.png)

For example, running the HPO function with the parameters to the right had been left to run for 91 hours (appendix 3) before the decision was made to reduce the number of parameters and breakdown the process to make things faster as adding each additional parameter increases the complexity exponentially.

Due to the above issues and not being able to process a full hyper-parameter optimization search, a lot of frustrations arose as a better classifier could undoubtedly have been found. Having to break this process down into smaller batches and trying to spot patterns and (in)consistencies in the outputs of a multitude of tests was a complex task that required focus and thorough note taking.

## References:
Brownlee, James - Machine Learning Mastery. 2019. A Gentle Introduction to k-fold Cross-Validation. [ONLINE] Available at: https://machinelearningmastery.com/k-fold-cross-validation/ [Accessed 15 May 2019].

McCaffrey, James - Visual Studio Magazine. 2013. Understanding and Using K-Fold Cross-Validation for Neural Networks. [ONLINE] Available at: https://visualstudiomagazine.com/articles/2013/10/01/understanding-and-using-kfold.aspx. [Accessed 14 May 2019].

Shruti Saxena – 2019. Precision vs Recall – Towards Data Science. [ONLINE] Available at: https://towardsdatascience.com/precision-vs-recall-386cf9f89488. [Accessed 21 May 2019]
Scikit-learn - 2019. sklearn.neural_network.MLPClassifier — scikit-learn 0.21.1 documentation. [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html. [Accessed 15 May 2019].

Scikit-learn - 2019. sklearn.model_selection.GridSearchCV— scikit-learn 0.21.1 documentation. [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. [Accessed 15 May 2019].

Scikit-learn - 2019. sklearn.preprocessing.StandardScaler— scikit-learn 0.21.1 documentation. [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html. [Accessed 15 May 2019].

