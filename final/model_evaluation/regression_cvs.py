
from final.hyperparameter_tuning.search_cv import CV_Model
from final.model_evaluation.regression_evaluation import reg_evaluation, get_ori_price
def find_train_best_model(classifier, parameters, x_train, y_train, x_test, y_test, data_name, price_split,cv_split=4,print_result=True):
    SearchCV = CV_Model(GridSearch=False)
    SearchCV.train_model(classifier, parameters, x_train, y_train, cv_split=cv_split)

    # train model with all train data again
    SearchCV.train_best_model(x_train,y_train)

    train_pred = SearchCV.pred_target(x_train)
    test_pred = SearchCV.pred_target(x_test)

    ori_train_price, ori_test_price, pred_train_price, pred_test_price = get_ori_price(y_train, train_pred,y_test,test_pred)

    print("Result of using %s"%data_name)
    return reg_evaluation(ori_train_price, ori_test_price, pred_train_price, pred_test_price,  # origin price
                   y_train, train_pred, y_test, test_pred,
                   price_split,print_result)

#     train_pred, test_pred
