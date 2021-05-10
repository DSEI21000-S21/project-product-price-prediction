Zhiying

# III. Transformation, Feature Selection, and Modeling: 30pts

**A. Did you transform, normalize, filter the data appropriately to solve your problem? Did you divide by max-min, or the sum, root-square-sum, or did you z-score the data? Did you justify what you did?**

Price - select >0,
text column - transform and vectorize
both categorical and vectorized text - normalize using min-max as some model requires normalization


**B. Did you justify normalization or lack of checking which works better as part of your hyper-parameters?**
`skip`


**C. Did you explore univariate and multivariate feature selection? (if not why not)**

Univaritate fature selection - Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator. Scikit-learn exposes feature selection routines as objects that implement the transform method. We tried We used sklearn library "SelectKBest" to select features according to the k highest scores. 

Recursive feature elimination: we tried Recursive feature elimination to select features by recursively considering smaller and smaller sets of features. Recursive Feature Elimination first uses ALL my features, fits a Linear Regression model, and then kicks out the feature with the smallest absolute value coefficient.


**D. Did you try dimension reduction and which methods did you try? (if not why not)**
 yes :
 - PCA
 - t-SNE for visualization
 - SVD 
 - k-Means
 - Latent dirichlet Allocation


**E. Did you include 1-2 simple models, for example with classification LDA, Logistic Regression or KNN?**
 yes, for basic models, we tried:
 - KNN regression
 - ridge regressor

**F. Did you pick an appropriate set of models to solve the problem? Did you justify why these models and not others?**
For basic models, we tried 
- KNN regression
- ridge regressor

For ensemble models, we tried
- lightBGM regressor
- random forest regression

In addition, we also tried neural networks. We also tried some other models, but obtained worse outcome


**G. Did you try at least 4 models including one Neural Network Model using Tensor-Flow or Pytorch?**
yes, we tried 5 models, which include neural networks

**H. Did you exercise the data science models/problems we described in the lectures showing what was presented?**
yes, we do train-test split before fitting the model. We tune the model hyperparameters to select the best hyperparameters for that specific model

**I. Are you using appropriate hyper-parameters? For example, if you are using a KNN regression are you investigating the choice of K and whether you use uniform or distance **weighting? If you are using K-means do you explain why K? If you are using PCA do you explore how many dimensions such as by looking at the eigenvalues?


# Metrics, Validation and Evaluation 20pts

A. Are you using an appropriate choice of metrics? Are they well justified? If you are doing classification do you show a ROC curve? If you are doing regression are you justifying the metric least squares vs. mean absolute error? Do you show both?

B. Do you validate your choices of hyperparameters? For example, if you use KNN or K-means do you use cross-validation to optimize your choice of parameters?

C. Did you make sure your training and validation process never used the training data?

D. Do you estimate the uncertainty in your estimates using cross-validation?  

E. Can you say how much you are overfitting?
