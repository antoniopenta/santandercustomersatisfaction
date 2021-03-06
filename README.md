# Santander Customer Satisfaction Prediction
Solution for the Kaggle competition https://www.kaggle.com/c/santander-customer-satisfaction

Project Structure

The solutions of the problem are described in the following notebooks (main folder):

DataExploration.ipynb : Notebook used to explore the data, to get a fist understanding of the features in order to select a data science strategy for further analysis.

FeatureSelection.ipynb: Notebook used to explore Boruta Method for features selection.

BinaryAndNumericFeatureAnalysis.ipynb:  Notebook used to explore the relevance of the binary features, and what kind of cleaning approach I should do for the numeric features. 

BinaryFeatureAnalysis.ipynb: Notebook used to further explore the binary features.

NumericFeatureAnalysis.ipynb: Notebook used to clean and normalize the no binary features.

CustomerSatisfactionClassification.ipynb: Notebook used to classify happy and unhappy customers

CustomerSatisfactionSantanderKaggle.pdf, Presentation of the results

In the submission folder there are scripts to test the results using the test set, the script run_strategy_15.py has obtained Private Score 0.802821 	Public Score 0.815378.
