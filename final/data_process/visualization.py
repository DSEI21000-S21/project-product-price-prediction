import matplotlib.pyplot as plt
import seaborn as sns
def visualization_str_column_attributes(df):
    for column in df.columns:
        if column == "train_id" or column == "price":
            pass
        unique_count = df[column].value_counts()
        unique_count.sort_values(ascending=False)[:min(20,  len(unique_count))].sort_values().plot(kind='barh')

        plt.xlabel(column)
        plt.ylabel("count")
        plt.title("Show frequency of the value in %s"%column)
        plt.show()

    # visualize price distribution
    df.price.plot.hist(bins=1000, alpha=0.5)
    plt.title("Item Price Distribution")
    plt.xlabel("price")
    plt.show()

    sns.boxplot(y='price', data=df)
    plt.show()
