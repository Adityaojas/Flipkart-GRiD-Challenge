# coding: utf-8

import numpy as np
# import matplotlib.pyplot as plt
import pickle
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# plt.switch_backend('agg')

def getdata():
    data = {}
    # read data and shuffle
    index=[i for i in range(14000)]
    random.shuffle(index)

    # f=open("./id_to_data","rb+")
    # # data=pickle.load(f)
    # j=1
    # while 1:
    #     try:
    #         data[j] = pickle.load(f)
    #         j+=1
    #     except EOFError:
    #         break
    f = open("./id_to_data", "rb+")
    data = pickle.load(f)
    data = data[index]

    # data = list(map(data.get, index))
    # print("hello")
    # data = data[index]
    # data=list(data.values())
    data_train=data[0:11000]
    data_test=data[11000:]

    with open("./data_test", 'wb+') as dd:
        pickle.dump(data_test, dd, protocol=4)

    f=open("./id_to_box","rb+")
    box=pickle.load(f)
    box=box[index]
    box_train=box[0:11000]
    box_test=box[11000:]

    with open ("./data_level", 'wb+' ) as dl:
        pickle.dump(box_test, dl, protocol = 4)
    return data_train,box_train,data_test,box_test


def plot_model(model_details):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_details.history['my_metric'])+1),model_details.history['my_metric'])
    axs[0].plot(range(1,len(model_details.history['val_my_metric'])+1),model_details.history['val_my_metric'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['my_metric'])+1),len(model_details.history['my_metric'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig("model_75.png")

