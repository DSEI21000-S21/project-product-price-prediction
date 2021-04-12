# Dataset: [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data?select=sample_submission_stg2.csv.zip)

## Data Description
- Files are tab-delimited (`.tsv`)

### Data Fields
- `train_id` or `test_id`: the id of the listing
- `name`: title of the product. 
    - this field have been clean by remove text that look like prices to avoid leakage
    - removed prices are represented as [rm]
- `item_condition_id`: the condition of the items provided by the seller
- `category_name`: category of the listing
- `brand_name`: 
- `shipping` - 1 if shipping fee is paid by seller and 0 by buyer
- `item_description` - the full description of the item. 
    - this field have been clean by remove text that look like prices to avoid leakage
    - removed prices are represented as [rm]
- `price`: price that the item was sold for. The unit is USD.
    - Include in the train data
    - Target for the test data
