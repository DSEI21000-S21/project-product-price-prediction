Zhiying

# III. Transformation, Feature Selection, and Modeling: 30pts

**A. Did you transform, normalize, filter the data appropriately to solve your problem? Did you divide by max-min, or the sum, root-square-sum, or did you z-score the data? Did you justify what you did?**

We transform, normalize and filter the data appropriately during our Exploratory Data Analaysis (EDA) process before solving our problem. In our "price" attribute, we have data where prices were reported less than $0, which are being filtered out during our data cleaning process. We specifically select "price" where it's above 0 dollar. 
In addtion, we also impute the missing value appropriately by replacing the missing values with either with "missing" for the brand and category_name attributes or "no description yet" for the item_description attribute. 

For text data, we apply text cleaning, text vecotorization, and text feature extraction as part of our EDA process. 

- In our text cleaning stage, we built a function called cleaning_text which does the following steps in sequence:
	1. Standardize all text to its lower case

	2. Use the re package to extract only characters and numbers and remove any special characters and emoji

	3. Use the NLTK stopwords package to remove stop word and non-alphabetical words

	4. Use the WordNetLemmatizer package to reduce word to its root form

	5. Use the string package to replace all punctuation by white space

This function takes an input column of a dataframe, perform all the five steps mentioned above sequentially, and output a cleaner version of the input feature.

- In the text-processing stage, we lower all uppercased text, remove all stop words, punctuation, and special characters for each of the input features. It's possible that any of these feature play a role in predicting price. For example, price may be higher or lower for records with more uppercased words or emoji. We do not want to lose any information from the processed text, hence before text preprocessing, we want to extract the uppercase counts, lowercase counts, percent of uppercase word in the corpus, percent of lowercase word in the corpus, average word length in a corpus, stop words counts, punctuation count, and special character count. We built a function called get_word_count and get_special_char_count which specifically extracted features mentions above.
	1. The get_word_count is performed on the "name", "description", and "c1", "c2", "c3", and "brand_name" attributes of the data.

	2. The get_special_char_count is performed on the "name", "item_description" attributes of the data
All of these extracted features will feed into the feature selection model as additional distinct features.

- In the text feature extraction stage, because the "brand_name" and the three item subcategories (c1,c2,c3) are encoded categorically. Before feeding them into our feature selection model, we performed CountVectorizer for feature extraction. The name and item_description attributes of our dataset contain a bulk of text strings. In order to extract salient features from these two columns, we use the TfidfVectorizer from the Sklearn package to perform text feature extraction. We limited the maximum features to be 15,000 for attribute with higher than 15,000 features. We remove the maximum feature limit for attributes with lower than 15,000 features and take however many features we are able to extract from that attribute.

- Data normalization is performed on both categorical and vectorized text columns. Min-Max normalization is performed on all original as well as extracted numerical attributes from the dataset except the train_id, item_condition_id, price, and shipping columns. If the difference of max and min is zero, then we drop that column.


**B. Did you justify normalization or lack of checking which works better as part of your hyper-parameters?**

For now, we only evaluate the model where we normalize all data before inputting to the training model. We will be conducting future analysis to look at whether removal of the data normalization step has any improvement on our model performance. 


**C. Did you explore univariate and multivariate feature selection? (if not why not)**

Yes, explore different feature selections models before feeding into our training model. 
- For univaritate fature selection model, we use the Sklearn SelectKBest module. This model selects the best features based on univariate statistical tests and use the F-value between target and feature for ranking. 
- For other feature selection models, we also try the Sklearn Recursive feature elimination (RFE) package. RFE selects features by recursively considering smaller and smaller groups of features by intiialy use all features to fit the model, and then sequentially kicks out features with the least weight. At each iteration, RFE removes 5% of the features. It also uses Ridge as the external estimator that assignes weights to features. 


**D. Did you try dimension reduction and which methods did you try? (if not why not)**

Yes, after text feature extractions, we have over 18,000 features. As we face dimensionality problem, we try various dimension reduction methods to reduce the dimensions. For examaple, we try Principal Componenet Analaysi (PCA), t-SNE, singular value decomposition (SVD), k-means clustering or Latent Dirichlet Allocation (LDA) to reduce the dimensions. 


**E. Did you include 1-2 simple models, for example with classification LDA, Logistic Regression or KNN?**

Yes, for the basic model, we try a non-parametric model called KNN regressor. We also try another linear parametric model called ridge regression on our dataset to predict our targeted variable. 


**F. Did you pick an appropriate set of models to solve the problem? Did you justify why these models and not others?**

Yes, we try various models where we see appropriate. 
- For basic models,  we try a KNN regression and ridge regression.
- For ensemble models, we try lightBGM regressor and random forest regression. LightBGM regressor is a gradient boosting algorithm where the previous model sets the target outcomes for the next model in order to minimize the error. Random forest regressor is a series of independent decision trees for predicting the target outcomes. 
- We also built our own neural network models with customized layers to try to predict the target outcome.
Of course, we also try some other models that are not listed here. We abandon those models because they obtained worse outcomes than the ones mentioned above. 


**G. Did you try at least 4 models including one Neural Network Model using Tensor-Flow or Pytorch?**

We try 5 different models, which include neural networks, KNN regression, ridge regression, lightBGM regression, and random forest regression.


**H. Did you exercise the data science models/problems we described in the lectures showing what was presented?**

Closely following the lecture and study materials, we do train-test split before fitting the model. We tune the model hyperparameters to select the best hyperparameters for that specific model.


**I. Are you using appropriate hyper-parameters? For example, if you are using a KNN regression are you investigating the choice of K and whether you use uniform or distance weighting? If you are using K-means do you explain why K? If you are using PCA do you explore how many dimensions such as by looking at the eigenvalues?**

For KNN regression, we use RandomizedSearchCV or GridSearchCV to look for the best n_neightbors and weight. 
For PCA, we also use eigenvalues and plotted the percent explained ratio graph to look for the elbow position when selecting the most optimal number of principal componenents to use for the model. 


# Metrics, Validation and Evaluation 20pts

**A. Are you using an appropriate choice of metrics? Are they well justified? If you are doing classification do you show a ROC curve? If you are doing regression are you justifying the metric least squares vs. mean absolute error? Do you show both?**

For our main evaluation metrics is RMSLE which stands for Root Mean Squared Logarithmic Error. Because our price distribution is skewed and impacted by outliers and RMSLE is very robust to eliminate the effect from outliers. Hence, RMSLE is our main evaluation metrics. We also evaluated the model using other metrics, which include Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), R Square (R^2), and maximum percentage difference. All metrics are being evaluated for the model at the end. 

**B. Do you validate your choices of hyperparameters? For example, if you use KNN or K-means do you use cross-validation to optimize your choice of parameters?**

To optmize our choice of parameter, we first use the RandomizedSearchCV model which include cross-validation to select the best hyperparameter, then we use models with the best hyperparameter to fit train the training set. 

**C. Did you make sure your training and validation process never used the training data?**
- yes, we did a train-test split before text vectorization, feature selection, hyparameter tuning, and model training. Becuase we only feed the training data into the training model, we are sure that there is no information leak. 

**D. Do you estimate the uncertainty in your estimates using cross-validation?**

**E. Can you say how much you are overfitting?**
