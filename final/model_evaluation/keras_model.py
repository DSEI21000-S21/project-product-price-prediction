from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam


class neural_network():
    def __init__(self, model_prefix="", lr = 0.001, nodes=[256,64,16], dropouts = [0.1,0.1,0.1]):
        self.model, self.model_name = self.get_model(lr = lr, nodes=nodes, dropouts = dropouts)
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

    def fit(self, trainData, trainLabel,n_epoch=5,epoch=1000, bs=128):


        for i in range(n_epoch):
            # np.random.shuffle(testIndex)
            history = self.model.fit(trainData, trainLabel,
                      initial_epoch= i * epoch,
                      epochs=(i + 1) * epoch,
                      batch_size=bs,
                      shuffle=True,
                      verbose=1)
        self.model.save(self.model_name + "_ep%d.h5" % (n_epoch))

        from contextlib import redirect_stdout
        model_structure_filename = self.model_name + "_Structure.txt"
        print("Save Model: ", model_structure_filename)
        with open(model_structure_filename, "w+") as f:
            with redirect_stdout(f):
                self.model.summary()


    def predict(self, data):
        return self.model.predict(data)[:, 0]



