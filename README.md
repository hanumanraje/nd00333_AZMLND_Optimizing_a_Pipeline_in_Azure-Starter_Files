# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
### Problem Statement
The dataset used for this ML pipeline is based on data collected by a Portuguese bank and contains variables related to clients, the last contact with
clients in the context of the current campaign, additional attributes related to the campaign, general economic/ social indicators and whether a client
has subscribed to a term deposit. The aim of this pipeline is to predict whether a particular client will subscribe to a term deposit or not given the
different attributes contained in the dataset. (Information about the dataset taken from https://archive.ics.uci.edu/ml/datasets/bank+marketing#)
### Explanation of the Solution
The best performing model was a Voting Ensemble model chosen by Azure AutoML with an accuracy of 91.74%.


## Scikit-learn Pipeline
In the Scikit-learn pipeline the data is obtained from an online source and then some simple data cleaning is performed which mainly consists of
encoding categorical variables.The algorithm used is a simple logistic regression. Hyperparameter tuning was performed for both the regularisation
inverse and the number of iterations.

### Benefits of the chosen parameter sampler
I chose a random parameter sampler which will provide a similar performance to a deterministic grid based approach, but with a significantly
smaller number of iterations.

### Benefits of the chosen early stopping policy
The bandit policy allows us to implement early stopping based on the performance relative in terms of performance compared to the best performing run
instead of either basing termination upon the performance relative to the median or whether a particular run is among a specified percentage
of worst performing runs. 

The explanations of the benefits of the chosen early stopping policy and the parameter sample are based upon the official Azure ML documentation on hyperparameter tuning: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters

## AutoML
The model chosen by AutoML is a Voting Ensemble model which combines 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier' and 'SGD'.
The only available hyperparameter shown in the logs is the weight of the different models used in the Voting Ensemble. AutoML chose higher weights for the first two classifiers
(rougly twice as high as as for the remaing ones).


## Pipeline comparison
The model chosen by AutoML turns out to perform better with an accuracy of about 91.74% as compared to about 91.05% of the logistic regression. This may be due to the fact that we used logistic regression for the scikit-learn pipeline which is a rather simple linear model and may hence not be able to fit the data as well as the ensemble model chosen by AutoML. In addition to this, hyperparameter tuning
will usually have not such a large impact on performance if we are using a rather simple model.The downside of a more sophisticated algorithm may be that
it could overfit the data.

## Future work
There are several possible ways to improve this experiment. For one thing we may want to try out **different models** in our scikit learn
pipeline. Logistic regression makes sense as a starting point, but given the no free lunch theorem, we should always try out different
models since there is no way to know ex-ante which model will perform best. 

Moreover, we may also want to reconsider our **performance measure** given that we are dealing with a highly **imbalanced dataset** where clients who subscribed to a term deposit account for only 11.20% of all cases. Rather than accuracy we may want to use a metric like AUC weighted offered by AutoML to obtain a more realistic assessment of model performance. Another way to deal with this issue would be to **collect
additional data** of positive cases and/ or to use resampling techniques to create a balanced dataset. (For more information see e.g. https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)
