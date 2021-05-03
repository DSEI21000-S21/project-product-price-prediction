# Product-Price-Prediction
**Team Member:** Jin Chen, Zhi Li, Juliana Ma, Zhiying Zhu
**Project Categorical:** Financial and Commerce



## **Motivation and Problem Description:** 

It is hard to scale a good price for a product. Small details can make big differences in pricing. The seasonal pricing trends of clothing can be heavily influenced by brand names. The descriptions of the products can also cause fluctuating prices. What determines the value of things? How should you price your products that can lead more purchasing from the customers? After analyzing this project, we might be able to find out the answers to these questions.
The data is collected from the sellers of the Mercari's marketplace. According to the user-inputted information such as text descriptions of their products, including details like product category name, brand name, and item condition, we will need to build an algorithm that automatically suggests the right product prices. 

**Dataset:** [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge/code) from Kaggle 



## Methodology: 

### Data Preprocessing

Data preprocessing involves a series of steps from handling missing values, text cleaning, one-hot-encoding on categorical attributes.
The step for text cleaning including
1. Make All Text Lower Case
2. Removing Punctuations
3. Removing Stopwords
4. Correct Spelling 
5. Lemmatization: converting a word to its base form.

### Data Visualization
- Show the pattern of the item description based on different item category
- Display some statistic of the data

### Feature Selection and Feature Extraction

*Feature Extraction*
- Text Vectorization -> converting clean text into numerical representation
  - Try different techniques and compare the outcome and pick the best method   
    - Count Vectorizer
    - Tfidf vectorizer
    - Hashing Vectorizer
    - Word2Vec

- Extract information about the string column
  - Number of Uppercase Words and Characters
  - Number of Words
  - Number of Characters
  - Average Word Length
  - Number of Punctuations
  - Number of Numerical Characters
  - Number of stop words


*Feature Selection*
There will be a lot of raw features, where there will be many noisy with in the raw features. Therefore, we need to apply multiple tehniques to select the best features  before pass into the model  to predict the price.
Some methods tried for feature  selection:
- Logistical Regression Model -> coeff
- Ensemble learning: Random Forest -> features importance
- Sequential backward Selection: using Recursive Backward Elimination in Scikit learn.


### **Experiment and Evaluation:** 
The dataset contains both string variables and categorical data, and we will have four experiments. The goal for each experiment is to predict product price and is a regression problem. 

We will build a regression model for each experiment, but will use different features for each model.

- Only use the categorical features
- Only use the string variables features
- Use the selected variable from both categorical features and string variables feature
- Use all features

All four experiments, we will use the model selection and hyperparameter tuning techniques described in the above section to find the best model. We have a price value available for the dataset, which will be used to compare with the four models’ prediction result by calculating the mean square error.
