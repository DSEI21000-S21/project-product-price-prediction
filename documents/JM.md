Juliana 

# Machine Learning Question: 20 pts

**A. Is the background context for the question stated clearly (with references)?**

Mercari, Japan’s biggest community-powered shopping app, knows this problem deeply. They’d like to offer pricing suggestions to sellers, 
but this is tough because their sellers are enabled to put just about anything, or any bundle of things, on Mercari's marketplace.
We need to build an algorithm that automatically suggests the right product prices with the provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition.    

The reference is from kaggle. 

**B. Is the hypothesis/problem stated clearly ("The What")?**

The hypothesis is description is the factor to be considered for determining  the price for the product. 


**C. Is it clear why the problems are important? Is it clear why anyone would care? ("The Why")**

Price prediciton is a tool to get the suggeted price automatically by using a good model or algorithm. 
It is very important for business owners to make business decisions.  


**D. Is it clear why the data were chosen should be able to answer the question being asked?**

The dataset is provided by Kaggle. The dataset includes over 100 millions rows and 8 attributes. It is big enough to get a representative overview of the market. 

**E. How new, non-obvious, significant are your problems? Do you go beyond checking the easy and obvious?**

How much the description can influce the pricing? People can take a guess, but no one can tell a proven answer easily. 

# Data Cleaning/Checking/Data Exploration: 20pts

**A. Did you perform a thorough EDA (points below included)?**

We performed EDA by doing Data Processing. We removed the invalid variables where the price is less or equal to 0.

**B. Did you check for outliers?**

We checked the outliers by visualizing the price distribution with box plots and histograms. In addition, we used the log price to handle the outliers better.

**C. Did you check the units of all data points to make sure they are in the right range?**

We checked the units of all data points and provided the price ranges. 

**D. Did you identify the missing data code?**

There are over 600,000 missing data in brand name and over 80,000 in item decription. 

**E. Did you reformat the data properly with each instance/observation in a row, and each variable in a column?**

We tried various ways to handle the missing data, such as replacing the missing data with the most frequent used words and removing them from the dateset.
We kept the missing values and replaced them with the specific decriptions.
"Missing" for brand name and "No description yet" for item description. 

**F. Did you keep track of all parameters and units?**



**G. Do you have a specific code for formatting the data that does not require information not documented (eg. magic numbers)?

There are no magic numbers. All the needed information and numbers are gotten from the provided dataset.

**H. Did you plot univariate and multivariate summaries of the data including histograms, density plots, boxplots?

**I. Did you consider correlations between variables (scatterplots)?

**J. Did you consider plot the data on the right scale? For example, on a log scale?

**K. Did you make sure that your target variables were not contaminating your input variables?

**L. If you had to make synthetic data was it a useful representation of the problem you were trying to solve?


# Visualization 10pts

**A. Do you provide visualization summaries for all your data and features?

**B. Do you use the correct visualization type, eg. bar graphs for categorical data, scatter plots for numerical data, etc?

**C. Are your axes properly labeled?

**D. Do you use color properly?

**E. Do you use opacity and dot size so that scatterplots with lots of data points are not just a mass of interpretable dots?

**F. Do you write captions explaining what a reader should conclude from each figure (not just saying what it is but what it tells you)?
