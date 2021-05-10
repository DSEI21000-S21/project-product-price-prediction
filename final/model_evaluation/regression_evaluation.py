import numpy as np
import pandas as pd
from final.model_evaluation.error_function import mape # Mean Absolute Percentage Error
from final.model_evaluation.error_function import get_max_min_percentage_diff

from final.model_evaluation.visualizations import plot_prediction_price
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

def get_ori_price(y_train, train_pred,y_test,test_pred):
    try:
        ori_train_price = np.expm1(y_train.values)
        ori_test_price = np.expm1(y_test.values)
    except:
        ori_train_price = np.expm1(y_train)
        ori_test_price = np.expm1(y_test)
    pred_train_price = np.expm1(train_pred)
    pred_test_price = np.expm1(test_pred)

    return ori_train_price,ori_test_price,pred_train_price, pred_test_price


def reg_evaluation(ori_train_price,ori_test_price,pred_train_price, pred_test_price, # origin price
                   y_train, train_pred,y_test,test_pred,
                   price_split, print_result = True):

    under_train = (ori_train_price <= price_split).nonzero()[0]
    above_train = (ori_train_price > price_split).nonzero()[0]
    under_test = (ori_test_price <= price_split).nonzero()[0]
    above_test = (ori_test_price > price_split).nonzero()[0]

    if print_result:
        print("-"*50)
        print("For All Price")

        print("Train Result ----------")
        get_max_min_percentage_diff(ori_train_price, pred_train_price)
        print("RMSLE is ", mse(y_train, train_pred, squared=False))
        print("R^2  is ", r2(y_train, train_pred))
        print("Mean Absolute Percentage Error is ", mape(ori_train_price, pred_train_price))
        print("Mean Absolute Error is ", mae(ori_train_price, pred_train_price))

        print("\nTest Result ----------")
        get_max_min_percentage_diff(ori_test_price, pred_test_price)
        print("RMSLE is ",mse(y_test, test_pred, squared=False))
        print("R^2 is ", r2(y_test, test_pred))
        print("Mean Absolute Percentage Error is ", mape(ori_test_price, pred_test_price))
        print("Mean Absolute Error is ", mae(ori_test_price, pred_test_price))

        print("-"*50)
        print("For price under $%d"%price_split)

        print("Train Result ----------")
        get_max_min_percentage_diff(ori_train_price[under_train], pred_train_price[under_train])
        knn_skb_select_train_msle = mse(y_train[under_train], train_pred[under_train], squared=False)
        print("RMSLE is ", knn_skb_select_train_msle)
        print("R^2  is ", r2(y_train[under_train], train_pred[under_train]))
        print("Mean Absolute Percentage Error is ", mape(ori_train_price[under_train], pred_train_price[under_train]))
        print("Mean Absolute Error is ", mae(ori_train_price[under_train], pred_train_price[under_train]))

        print("\nTest Result ----------")
        get_max_min_percentage_diff(ori_test_price[under_test], pred_test_price[under_test])
        knn_skb_select_test_msle = mse(y_test[under_test], test_pred[under_test], squared=False)
        print("RMSLE is ", knn_skb_select_test_msle)
        print("R^2 is ", r2(y_test[under_test], test_pred[under_test]))
        print("Mean Absolute Percentage Error is ", mape(ori_test_price[under_test], pred_test_price[under_test]))
        print("Mean Absolute Error is ", mae(ori_test_price[under_test], pred_test_price[under_test]))

        print("-"*50)
        print("For price above $%d"%price_split)

        print("Train Result ----------")
        get_max_min_percentage_diff(ori_train_price[above_train], pred_train_price[above_train])
        knn_skb_select_train_msle = mse(y_train[above_train], train_pred[above_train], squared=False)
        print("RMSLE is ", knn_skb_select_train_msle)
        print("R^2  is ", r2(y_train[above_train], train_pred[above_train]))
        print("Mean Absolute Percentage Error is ", mape(ori_train_price[above_train], pred_train_price[above_train]))
        print("Mean Absolute Error is ", mae(ori_train_price[above_train], pred_train_price[above_train]))

        print("\nTest Result ----------")
        get_max_min_percentage_diff(ori_test_price[above_test], pred_test_price[above_test])
        knn_skb_select_test_msle = mse(y_test[above_test], test_pred[above_test], squared=False)
        print("RMSLE is ", knn_skb_select_test_msle)
        print("R^2 is ", r2(y_test[above_test], test_pred[above_test]))
        print("Mean Absolute Percentage Error is ", mape(ori_test_price[above_test], pred_test_price[above_test]))
        print("Mean Absolute Error is ", mae(ori_test_price[above_test], pred_test_price[above_test]))

    plot_prediction_price(ori_train_price[under_test], pred_train_price[under_test],
                          title="Predict Price for Item in Train Set with Price <= %d" %price_split)
    plot_prediction_price(ori_train_price[above_test], pred_train_price[above_test],
                          title="Predict Price for Item in Train Set with Price > %d" %price_split)

    plot_prediction_price(ori_test_price[under_test], pred_test_price[under_test],
                          title="Predict Price for Item in Test Set with Price <= %d" %price_split)
    plot_prediction_price(ori_test_price[above_test], pred_test_price[above_test],
                          title="Predict Price for Item in Test Set with Price > %d" %price_split)


    all_temp = []
    for indexes in [None, under_train, above_train]:
        if indexes is None:
            oy,py = ori_train_price, pred_train_price
            oly, ply = y_train, train_pred
        else:
            oy, py = ori_train_price[indexes], pred_train_price[indexes]
            oly, ply = y_train[indexes], train_pred[indexes]
        all_temp.append([*get_max_min_percentage_diff(oy, py),
                     mse(oly, ply, squared=False),
                     r2(oly, ply),
                     mape(oy, py),
                     mae(oy, py)
                     ])

    for indexes in [None, under_test, above_test]:
        if indexes is None:
            oy, py = ori_test_price, pred_test_price
            oly, ply = y_test, test_pred
        else:
            oy, py = ori_test_price[indexes], pred_test_price[indexes]
            oly, ply = y_test[indexes], test_pred[indexes]
        all_temp.append([*get_max_min_percentage_diff(oy, py),
                         mse(oly, ply, squared=False),
                         r2(oly, ply),
                         mape(oy, py),
                         mae(oy, py)
                         ])

    result_df = pd.DataFrame(np.array(all_temp), columns=['Max Percentage Diff','Min Percentage Diff','RMSLE','R^2','MAPE','MAE'],
                             index=['All Train', 'Train with Price <= %d'%price_split,'Train with Price > %d'%price_split,
                                    'All Test', 'Test with Price <= %d' % price_split,
                                    'Test with Price > %d' % price_split,
                                    ])
    return result_df
