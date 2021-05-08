import numpy as np

def mape(actual, pred):
    # mean absolute percentage error
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def get_max_min_percentage_diff(ori, pred):
    z = np.abs((ori - pred) / ori)
    max_ind = np.argmax(z)
    print("Max Percentage Difference: ", z.max())
    print("With Original Price %.2f, Predict Price %.2f: "%(ori[max_ind],pred[max_ind]))

    min_ind = np.argmin(z)
    print("Min Percentage Difference: ", z.min())
    print("With Original Price %.2f, Predict Price %.2f: "%(ori[min_ind],pred[min_ind]))
