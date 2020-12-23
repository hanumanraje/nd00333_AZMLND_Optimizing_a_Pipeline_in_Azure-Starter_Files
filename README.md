# Optimizing an ML pipeline in Azure

## Summary of the problem statement

In this project, we want to create and optimize ML pipeline. To do this, we are using the [UCI Machine Learning Repository: Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) to predict whether the client will subscribe a term deposit. 

## How the problem was solved

We are creating a tabular dataset from the link mentioned. After preprocessing we are using it to train a sci-kit learn logistic regression model and then we are optimizing its hyperparemeters `C` and `max-iter` through Azure ML's Hyperdrive. After finding the most optimal parameters, we are using the exact same dataset and running an AutoML experiement. AutoML then tries to identify which algorithm provide best accuracy with given configuration. After these results we will compare and evaluate which process provides beter models.

## Explanation of various factors

### Data
The data is taken from the UCI repository: [UCI Machine Learning Repository: Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 


#### Attribute Information:

Input variables:  
bank client data:  
1 - age (numeric)  
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')  
5 - default: has credit in default? (categorical: 'no','yes','unknown')  
6 - housing: has housing loan? (categorical: 'no','yes','unknown')  
7 - loan: has personal loan? (categorical: 'no','yes','unknown')  

related with the last contact of the current campaign:  
8 - contact: contact communication type (categorical: 'cellular','telephone')  
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')  
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.  

other attributes:  
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  
14 - previous: number of contacts performed before this campaign and for this client (numeric)  
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')  

social and economic context attributes  
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)  
17 - cons.price.idx: consumer price index - monthly indicator (numeric)  
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)  
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)  
20 - nr.employed: number of employees - quarterly indicator (numeric)  
  
Output variable (desired target):  
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

Note: The above information about the features is taken from [UCI Machine Learning Repository: Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)



### Architecture
The  architecture of the solution is as follows:
![System Architecture](https://video.udacity-data.com/topher/2020/September/5f639574_creating-and-optimizing-an-ml-pipeline/creating-and-optimizing-an-ml-pipeline.png "System Architecture")
We are fetching the dataset from this [link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and creating a tabularDataset using tabularDatasetFactory. Once we have the tabular dataset factory we will divide our progress in two ways. 

First, We will train a sci-kit learn logistic regression model and optimize its hyperparameters using Azure ML's Hyperdrive. We will perform several iterations of this while randomly sampling values of `C` and `max-iter` from the hyperdrive configuration. After training we will note down its results. Then we will use the same dataset, create an AutoML configuration and submit the experiment. The AutoML experiement will then go through various models with a number of hyperparameters and try to identify the best model that can most accurately predict the outcome.

### Algorithm and Hyperparameters
The model that we are training through Hyperdrive is the sci-kit learn Logistic Regression. Logistic regression is one of the supervised classification machine learning algorithms where we take the input and try to classify it into two outcomes. Some of the examples of usage of logistic regresiona are, dog classification, yes or no problems, will a student pass or fail in an exam, etc. 

The hyperparameter's we will be optimizing are `C` and `max-iter`.

**C** 
float, default=1.0
This paremeter determines the inverse of the regularization strength. Smaller the value of this parameter, bigger is the regularization.
In my approach, I have uniformly selected the value of c from 0 to 1.

**max_iter**
int, default=100
This paremeter determines the maximum amount of iterations to do before converging.
In my approach, I am randomly selecting a number from 0 to 100.


### Why RandomParameterSampler?
This class is used to define random sampling over the search space of the hyperparemeter we are trying to optimize. The random sampling statistically provides much better search results and require less resources. I ran the experiement multiple times using random sampling and each time it converged quickly with the same accuracy. Bayesian sampling intelligently picks new set of hyperparemeters based on previous runs but is more resourse intensive. Grid sampling also evaluates all possibilites in the search space. Hence, it utilizes more resourses. Hence, RandomParameterSamples best fits our use case due to the time constraints of the VM.

### Why BanditPolicy?
BanditPolicy first trains a model and identifies its accuracy. Once it has a base value, it trains the next model. If the accuracy of the new model is not within the slack factor specified then it cancels the run. If the accuracy is better, then the accuracy of the new model becomes the new benchmark. This allows us to eliminate all the lower accuracy models that are not performing well. MedianStopping policy calculates a running average of primary metric. While TruncationSelectionPolicy cancels a percentage of runs with search space having less performance. BanditPolicy helps us to relate the new runs with previous and effectively eliminate all the runs that are performing less than the current run. Hence, is more effective for this problem. 


## Comparison of the outputs
### Hyperdrive

 - Finished much quickly as it was training only one model over the search space of hyperparemeters.
 - We had to do the data preprocessing on our own.
 - Using Hyperdrive, sci-kit learn's Logistic regression achieved 90.74% accuracy.
 - Most optimal values for the hyperparemters we optimized are
	 - C - 0.2973202492447354
	 - max-iter - 16
 - It looks like only a maximum of 16 iterations was enough to produce an accuracy of 90% with an inverse regularization of 0.297.

### AutoML

 - AutoML training process took longer as it trained more than 50 models!
 - It intelligently performed the data preprocessing, featurization and balancing of the classes in the dataset.
 - After doing the the AutoML warned us that the class when the client will subscribe a term deposit. out of 32k samples only 3k samples indicate this. Which means that our model can suffer from Accuracy paradox. In order to avoid this, AutoML smartly augmented the features for this class.
 - During training we can observe that MaxAbsScalar SGD occurs multiple times. What AutoML trying to do in this case is it takes into account its previous runs and tries to optimize its hyperparemeters. And then it tries to see if the current hyperparemeters worked better than the previous one.
 - The information displayed also keeps the track of the maximum accuracy observed so far so that it becomes easier to see which models perform better as compared to previous models.
 - Out of all the models AutoML trained, the ensamble techniques like voting ensables and stack ensables outperformed every other model including the most optimal logistic regression model we achieved through Hyperdrive. The reason for this is ensable techniques like these take multiple models into account and accumulate their results. Hence their performance was much better.
 - Details of the best model (Voting Ensamble):
 	 - The function to explain the AutoML model was suggested by the reviewer. Thank you so much for that, It prints detailed information about the different ML models used in  the ensambling and prints their hyperparemeters
	 - The voting ensable combined ML various models like xgboostclassifier, lightgbmclassifier, sparsenormalizer, sgdclassifierwrapper, maxabsscaler, standardscalerwrapper and the existing prefittedsoftvotingclassifier.
	 - There are multiple occurances of these models with different sets of hyperparameters. This is why you see classifiers like XGBoost in the explanation of the model. And as an objective, there's logistic regression in one case while binary logistic classification in another. Other parameters like `max_child_weight`,`eta` vary per occurance of the model.
	 - Each model contributes to the combined final score.

## Scope of improvement

 - Please specify the problem that we are trying to solve instead of just specifying the dataset directly. The Azure's Hyperdrive and AutoML are techniques to solve such problems and are not the problem itself.
 - We can try any machine learning algorithms other than logistic regression and see how the data behaves with several combinations hyperparameters. We can also try out several robust models like XGBoost and LightGBM.. It'll help students practically on how we should go about choosing the model.
 - We can try other preprocessing techniques including the clean_data function in train.py like sci-kit learns imputer which can help students increase their ETL knowledge.
 - We can probably collect more data and vary the cross_validations and change the experiment_timeout so that AutoML tries out several models and preprocessing steps to come up with the model having the best performance.

## Proof of Cluster Cleanup
I deleted the cluster from the ML Studio's compute section.
![clean](./clean.png "Cluster deletion")
