import os
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.models import load_model
from final.helper.save_data import save_np_file, save_model_structure,save_model
import numpy as np
from final.model_evaluation.regression_evaluation import reg_evaluation, get_ori_price

class neural_network():
    def __init__(self, model_name = "", model_prefix="",lr = 0.001,
                 nodes=[256,64,16], dropouts = [0.1,0.1,0.1]):
        if os.path.exists(model_name):
            print("loading model")
            self.model = load_model(model_name)
            self.model_name = model_name[:model_name.rindex('_')]
        else:
            self.model, self.model_name = self.get_model(lr=lr, nodes=nodes, dropouts=dropouts)
            self.model_name = model_prefix + '_' + self.model_name

    def get_model(self, lr = 0.001, nodes=[256,64,16], dropouts = [0.1,0.1,0.1]):
        model_name = "NN"
        model = Sequential()
        for node, dropout in zip(nodes,dropouts):
            model.add(Dense(node, activation='relu'))
            model_name += "_%d" % node
            if 0 < dropout < 0.5:
                model.add(Dropout(dropout))
                model_name += "_dr%.0e" % dropout

        model.add(Dense(1)) # predict price

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
        model_name += "_lr%.0e" % lr

        return model, model_name

    def fit(self, trainData, trainLabel,testData, testLabel, n_epoch=5,epoch=1000, bs=128):

        loss = []
        for i in range(n_epoch):
            # np.random.shuffle(testIndex)
            history = self.model.fit(trainData, trainLabel,
                      initial_epoch= i * epoch,
                      epochs=(i + 1) * epoch,
                      validation_data=(testData, testLabel),
                      batch_size=bs,
                      shuffle=True)
            loss.append([history.history['loss'],history.history['val_loss']])
            if (i+1)%5 == 0:
                save_model(dir=self.model_name, filename=self.model_name + "_ep%d.h5" % (i+1),
                           model=self.model)

                save_np_file(dir=self.model_name,
                             filename="select_k_best_y_train_ep%d.npy" % (i+1),
                             data=self.predict(trainData))
                save_np_file(dir=self.model_name,
                             filename="select_k_best_y_test_ep%d.npy" % (i+1),
                             data=self.predict(testData))
        # save_model_structure(dir= self.model_name, filename=self.model_name + "_structure.txt", model = self.model)

        save_model(dir = self.model_name, filename=self.model_name + "_ep%d.h5" % (n_epoch),
                   model=self.model)


        save_np_file(dir = self.model_name, filename=self.model_name + "_loss.npy",data=np.array(loss))




    def predict(self, data):
        return self.model.predict(data)[:, 0]

    def evaluation(self,y_train, train_pred, y_test, test_pred, price_split, print_result= True):
        print("Result of using %s" %  self.model_name)
        ori_train_price, ori_test_price, pred_train_price, pred_test_price = get_ori_price(y_train, train_pred, y_test,
                                                                                           test_pred)


        return reg_evaluation(ori_train_price, ori_test_price, pred_train_price, pred_test_price,  # origin price
                       y_train, train_pred, y_test, test_pred,
                       price_split,print_result)

