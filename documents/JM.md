Juliana 

# Machine Learning Question: 20 pts

**A. Is the background context for the question stated clearly (with references)?**

Mercari, Japan’s biggest community-powered shopping app, knows this problem deeply. They’d like to offer pricing suggestions to sellers, 
but this is tough because their sellers are enabled to put just about anything, or any bundle of things, on Mercari's marketplace.
We need to build an algorithm that automatically suggests the right product prices with the provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition.    

The reference is gotten from Kaggle. 

**B. Is the hypothesis/problem stated clearly ("The What")?**

How should you determine the price for your product?
The hypothesis is description is the factor to be considered for determining  the price for the product. 


**C. Is it clear why the problems are important? Is it clear why anyone would care? ("The Why")**

Price prediciton is a tool to get the suggeted price automatically by using a good model or algorithm.  
For the business owners, price prediction is essential in the success of the operation as pricing defines the extent of profit.
Price prediction is also a key factor for the clientele in their purchase decision because proper pricing is within their expectations.
Furthermore, price prediction is the mission of the Data Scientists. Their quality of work is directly related to the price decision of the business executive.


**D. Is it clear why the data were chosen should be able to answer the question being asked?**

The dataset is provided by Kaggle. The dataset includes over 100 millions rows and 8 attributes. It is big enough to get a representative overview of the market. 

**E. How new, non-obvious, significant are your problems? Do you go beyond checking the easy and obvious?**

How much the description can influce the pricing? People can take a guess, but no one can tell a proven answer easily. 
Therefore, our problem is significant.

# Data Cleaning/Checking/Data Exploration: 20pts

**A. Did you perform a thorough EDA (points below included)?**

We performed a thorough EDA with the following steps:
1. Data Processing -- removed the invalid variables where the price is less or equal to 0.
2. Visualize the distributions for each column
3. Cleaned and formated the text 
4. Extracted features
5. Normalized the numeric data 
6. Checked the correlations between each column

**B. Did you check for outliers?**

We checked the outliers by visualizing the price distribution with box plots and histograms. In addition, we used the log price to better handle the outliers.

**C. Did you check the units of all data points to make sure they are in the right range?**

We checked the units of all data points and provided the price ranges. 

**D. Did you identify the missing data code?**

There are over 600,000 missing data in brand name and over 80,000 in item decription. 

**E. Did you reformat the data properly with each instance/observation in a row, and each variable in a column?**

We tried various ways to handle the missing data, such as replacing the missing data with the most frequent used words and removing them from the dateset.
We kept the missing values and replaced them with the specific decriptions.
"Missing" for brand name and "No description yet" for item description. 

**F. Did you keep track of all parameters and units?**

We tracked all parameters and units to make sure the results could be used for the next step.

**G. Do you have a specific code for formatting the data that does not require information not documented (eg. magic numbers)?

There are no magic numbers. All the needed information and numbers are gotten from the provided dataset.

**H. Did you plot univariate and multivariate summaries of the data including histograms, density plots, boxplots?**

There are box plots for showing the price outliers and distributions among categories with price. 
There are histograms with regular price and log price to compare the clearity of showing the price distributions. 
There are density plots for showing the smoothed price distribution of points along the numeric axis. 

**I. Did you consider correlations between variables (scatterplots)?**

We considered correlations between variables.
We created scatterplots to display the correlations between the actual price and predicted price with different price ranges. 

**J. Did you consider plot the data on the right scale? For example, on a log scale?**

For visualizing the price distribution, we used the log price for both boxplots and histograms. 
That makes the plots more limpid readable.

**K. Did you make sure that your target variables were not contaminating your input variables?**

All results that we got were computed from the input variables. We are sure that our target variables were not contaminating input variables.  

**L. If you had to make synthetic data was it a useful representation of the problem you were trying to solve?**

We made the synthetic data from the original dataset by using the sample function.
The synthetic data is a random sample of items from the original dataset. Therefore, it was a useful representation of the problem.



# Visualization 10pts

**A. Do you provide visualization summaries for all your data and features?**

We provided visualization summaries for all our data and features. 

**B. Do you use the correct visualization type, eg. bar graphs for categorical data, scatter plots for numerical data, etc?**

The distributions are shown with histograms. The correlations are shown with heatmaps. The categrical groupings are shown with clustering plots. 
The outliers are shown with box plots. The rankings are shown with bar charts.  In addition, the relationships between the variables are shown with scattor plots. 

**C. Are your axes properly labeled?**

All visuals contain proper labels for each axis. We can clearly see what does the axis represent for. 

**D. Do you use color properly?**

The colors are assigned to each group of data on the visuals. In another word, by seeing the colors, we are able to distinguish the groups.

**E. Do you use opacity and dot size so that scatterplots with lots of data points are not just a mass of interpretable dots?**

The dots are beautifully shown on the clustering plot. On the plot, each color represents a category. 
By seeing the dot, we would know where the specific categoty belongs to. The dots do not cover each other, so we can see a clear separation of groups of the categories.  

**F. Do you write captions explaining what a reader should conclude from each figure (not just saying what it is but what it tells you)?**

The visualizations can provide good insights of the information. By look at the visualizations, the reader can get a better idea on how should him/her determine the price for his/her products.
