import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def visualization_str_column_attributes(df):
    for column in df.columns:
        if column == "train_id" or column == "price":
            continue
        unique_count = df[column].value_counts()
        unique_count.sort_values(ascending=False)[:min(20,  len(unique_count))].sort_values().plot(kind='barh')

        plt.xlabel(column)
        plt.ylabel("count")
        plt.title("Show frequency of the value in %s"%column)
        plt.show()

    # # visualize price distribution
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # df.price.plot.hist(bins=1000, alpha=0.5, ax=ax1)
    # sns.boxplot(y='price', data=df, ax= ax2)
    #
    # ax1.title.set_text("Item Price Distribution")
    # ax1.set_xlabel("price")
    # plt.show()


def visualization_numeric_distribution(df,numeric_column):
    for column in numeric_column: #df.select_dtypes(include=np.number).columns:
        # visualize price distribution
        fig, (ax1, ax2) = plt.subplots(1, 2)
        df.price.plot.hist(bins=1000, alpha=0.5, ax=ax1)
        sns.boxplot(y=column, data=df, ax=ax2)

        # ax1.title.set_text("Item Price Distribution")
        ax1.set_xlabel(column)
        fig.suptitle("Item %s Distribution" % column)
        plt.show()


def display_correlation_matrix(df):
    numeric_col = df.select_dtypes(include=np.number).columns.tolist()
    numeric_col.remove('train_id')
    numeric_col.remove('price')

    # loop for correlation matrix
    corrs = []
    i = len(numeric_col) // 4

    corr1 = df[numeric_col[:i] + ['price']].corr()
    corr2 = df[numeric_col[i:2 * i] + ['price']].corr()
    corr3 = df[numeric_col[2 * i:3 * i] + ['price']].corr()
    corr4 = df[numeric_col[3 * i: ] + ['price']].corr()
    all_corrs = [corr1, corr2, corr3, corr4]
    for corr in all_corrs:
        corrs.extend(list(zip(corr['price'][:-1], corr.columns[:-1])))

        matrix = np.triu(corr)
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, mask=matrix, cmap='coolwarm')
        plt.title("Correlation Coefficient Matrix")
        plt.show()

    # sorting coeff
    sorted_list = list(zip(*sorted(corrs, key=lambda t: abs(t[0]), reverse=True)))
    return all_corrs, sorted_list

def plot_sort_corr_coeff(sorted_list, num_to_display = 10):
    # plot top features
    fig, ax = plt.subplots()
    x = np.arange(len(sorted_list[1][:num_to_display]))  # the label locations
    ax.bar(x, sorted_list[0][:num_to_display])
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_list[1][:num_to_display])
    plt.xticks(rotation=-45)
    plt.title("Correlation with price", fontdict={'fontsize': 20})
    plt.show()

def plot_pair_plot(df, columns):
    sns.pairplot(df[columns])
    plt.show()

