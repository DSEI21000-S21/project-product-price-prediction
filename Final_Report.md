# Product-Price-Prediction
**Team Members:** Jin Chen, Zhi Li, Juliana Ma, Zhiying Zhu

## **Motivation and Problem Description** 

As a person just entering the retail industry, how can they quickly learn about the market? 
How could they scale a good price for their product? If they price too high, people will not pay for it, 
but if they price too low, people could either question about their product or they will not have enough profit. 
Beside the retail person, the business owner also have the need for understanding the market, to know how their 
competitors pricing the similar product, and use that information to do their business strategies planning. 
So the main question here is what affect the product price and how could we determine the best price range for a product 
so it will not off from the exist market. 

There are many common senses on the things that could affect the price, such as clothes price can vary by the season and 
the brand, even the descriptions of the products can also cause fluctuating prices. 
To better understand these question and explore the answer, we look into the dataset provide by [Mercari](https://www.mercari.com),  
Japan’s biggest community-powered shopping app. The dataset contains over 1.4 millions of product records, where each records 
consists the seller inputted information of the product they are selling, including the item name, item conditions, 
brand name, item categories, shipping status, and item descriptions. 
It is a great dataset for us to explore the potential answers to our question, for many reasons. 
First, online selling is more demand in today's world and sellers not only sell the new product but also 
the remanufactured product or second-hand product, and this dataset could the product from a large diverse range. 
Second, the sellers from the Mercari, range from individuals just enter the retail field to experience business, 
so we can have a more general analysis of the market.
Third, the dataset contained large number of records that can provide a great representative overview of the market. 

Based on the dataset, we would like to know if the item desciption and other fields that dataset provided can really 
help people determine the price range of the product, and if so, how accurate that predict price range will be. 
To find the answer for our problem, we explore different data process and analysis algorithm and machine learning models 
to build the pipeline for our price prediction. The general pipeline of our process is show in the below figure, where the
color of each components reflect on the works of each team member. 

![Product Prediction Pipeline](./image_assets/Pipeline.png)
