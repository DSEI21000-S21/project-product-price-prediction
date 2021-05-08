import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_prediction_price(origin, predict, title):
    _, ax = plt.subplots(figsize = (15,5))
# matplotlib.rcParams['figure.figsize'] = (10, 10)
    ax.scatter(x = range(0, origin.size), y=origin, c = 'blue', label = 'Actual Price', alpha = 0.2)
    ax.scatter(x = range(0, predict.size), y=predict, c = 'red', label = 'Predicted Price', alpha = 0.2)

    plt.title(title, fontdict = {'fontsize' : 20})
    plt.xlabel('Observations')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

def plot_loss(loss, title):
    plt.plot(loss)
    plt.ylabel('msle')
    plt.xlabel('epoch')
    plt.title(title)
    plt.show()
#
# def visualize_model_feature_importances(trained_model, feature_names, title = None):
#     try:
#         importances = trained_model.feature_importances_
#     except:
#         importances = np.sum(abs(trained_model.coef_), axis=0)
#     indices = np.argsort(importances, )
#
#     # plot feature importance
#     if title:
#         plt.title(title)
#     plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#     plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
#     plt.xlabel('Relative Importance')
#     plt.show()
#
#     feature_importances = [(feature_names[i], importances[i]) for i in indices[::-1]]
#
#     return feature_importances
#
#
# def visualize_2d_cluster_with_legend(classname, feature1, feature2, X_names, y_names,
#                                      X_train, X_test, y_train, y_test, y_train_pred, y_test_pred,
#                                      legend=True, title=None):
#     if len(y_train.shape) > 1:
#         y_train = np.argmax(y_train, axis=1)
#         y_test = np.argmax(y_test, axis=1)
#     if len(y_train_pred.shape) > 1:
#         y_train_pred = np.argmax(y_train_pred, axis=1)
#         y_test_pred = np.argmax(y_test_pred, axis=1)
#
#     train_df = pd.DataFrame(X_train, columns=X_names)
#     train_df['%s_true' % classname] = list(map(lambda x: y_names[x], y_train))
#     train_df['%s_pred' % classname] = list(map(lambda x: y_names[x], y_train_pred))
#
#     test_df = pd.DataFrame(X_test, columns=X_names)
#     test_df['%s_true' % classname] = list(map(lambda x: y_names[x], y_test))
#     test_df['%s_pred' % classname] = list(map(lambda x: y_names[x], y_test_pred))
#
#     fig, axs = plt.subplots(2, 2)
#     sns.scatterplot(data=train_df, x=feature1, y=feature2, ax=axs[0, 0], hue='%s_true' % classname, palette="deep")
#     sns.scatterplot(data=train_df, x=feature1, y=feature2, ax=axs[0, 1], hue='%s_pred' % classname, palette="deep")
#
#     sns.scatterplot(data=test_df, x=feature1, y=feature2, ax=axs[1, 0], hue='%s_true' % classname, palette="deep")
#     sns.scatterplot(data=test_df, x=feature1, y=feature2, ax=axs[1, 1], hue='%s_pred' % classname, palette="deep")
#
#     axs[0, 0].title.set_text('Train - True Class')
#     axs[0, 1].title.set_text('Train - Predict Class')
#     axs[1, 0].title.set_text('Test - True Class')
#     axs[1, 1].title.set_text('Test - Predict Class')
#
#     if title:
#         plt.title(title)
#
#     if legend:
#         handles, labels = axs[0, 0].get_legend_handles_labels()
#         fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper center')
#
#     # fig.tight_layout()
#     axs[0, 0].get_legend().remove()
#     axs[0, 1].get_legend().remove()
#     axs[1, 0].get_legend().remove()
#     axs[1, 1].get_legend().remove()
#     plt.show()
