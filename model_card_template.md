# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model used for this classification task is the RandomForestClassifier from Sklearn.\
After some hyper-parameter tuning I found that using n_estimators=267, max_depth=17 for settings made the best model.
I also used a random state of 42 for consistent testing. 
## Intended Use
The intended use of this model is to predict whether someone earns over 50k in salary. This is based off of census data.\
The audience usage is for students/academic purposes only.
## Training Data
I could not find a source for the data in the starter kit assigned, so I'm unable to confirm what census the 
data is taken from. I also am unable to account for how values of the individual features were figured or what they
truly represent. However, based on the names and data types I will say what I believe them to represent. 
\
\
Data exploration can be found in a jupyter notebook titled: data_exploration.ipnyb.\
There are 6 continuous and 8 categorical features, with a final column, "salary", as the label feature.\
Continuous features:
- age: age of person surveyed
- fnlgt: Unknown, probably an acronym or unique identifier
- education_num: represents education level numerically
- capital-gain: capital gains received in the past year
- capital-loss: capital losses received in the past year
- hours-per-week: average amount of hours worked per week in the past year

Categorical features:
- workclass: type of work, and level examples are 'State-gov', 'Local-gov', and 'Private'
- education: level of education achieved
- marital-status: marital status of individual or involvement of partner
- occupation: generic job occupation categories, 'Sales', 'Adm-clerical', etc.
- relationship: description of your current relationship
- sex: individuals sex
- native-country: individuals native country

Label feature:
- salary: '>50K' and '<=50k'

## Evaluation Data
The cleaned data was split using train_test_split from sklearn. It uses 20% for the test data.
## Metrics
To measure model performance I used Precision, Recall, and F1 scores. They are as follows:
- Precision: 0.7913 
- Recall: 0.5913
- F1: 0.6769

## Ethical Considerations
As data collection method is unclear, biases towards population groups may be present.
## Caveats and Recommendations
As I cannot confirm the original source of data, nor its collection methods this model and its 
accompanying data should be considered as a thought experiment and not used as fact or validity.
It is recommended to only be used as an example of what an ML Classifier can do.