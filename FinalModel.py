from PIL import Image
import numpy as np
import pickle
import json
import pandas as pd

train_df = pd.read_csv("training.csv")
im_name = train_df.image_name
length = len(im_name)

open("./images.txt",'w').close()
open("./bounding_boxes.txt", 'w').close()

fh = open("./images.txt", 'a+')

for i in range(length):
    fh.write(str(i+1) + " " + im_name[i] + "\n")

fh.close()

test_df = pd.read_csv("test.csv")
im_name = test_df.image_name
length = len(im_name)

open("./images_test.txt",'w').close()

fh = open("./images_test.txt", 'a+')

for i in range(length):
    fh.write(str(i+1) + " " + im_name[i] + "\n")

fh.close()

x1 = train_df.x1
x2 = train_df.x2
y1 = train_df.y1
y2 = train_df.y2

fh = open("./bounding_boxes.txt", 'a+')

for i in range(length):
    fh.write(str(i+1) + " " + str(x1[i]) + " " + str(y1[i])+ " " + str(x2[i])+ " " + str(y2[i]) + "\n")

fh.close()



def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

id_to_data={}
id_to_size={}
i=0
open('./id_to_data', 'w').close()
open('./id_to_size', 'w').close()
open('./id_to_box', 'w').close()
with open("./images.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,path=line.split(" ",1)
        image=Image.open("./images/"+path).convert('RGB')
        id_to_size[int(id)]=np.array(image,dtype=np.float32).shape[0:2]
        # print(np.size(image, 0))
        # print(np.size(image, 1))
        # print(np.size(image, 2))
        image=image.resize((220,220))
        image=np.array(image,dtype=np.float32)
        image=image/255
        image=Normalize(image,[0.485,0.456,0.406],[0.229,0.100,0.225])
        id_to_data[int(id)]=image
        print(id)
        # if int(int(id)/11788) == int(id)/11788:
        # print(int(int(id)/11788))
        # fi = open("./id_to_data", "a+")
        # fi.write(json.dumps(image.tolist()))

        # pickle.dump(image, fi, protocol=4)
        # image = np.array2string(image)
        # # fi.write(image)
        # # print(i)
        # fi.close()
        # print(np.size(image,0))
        # print(np.size(image, 1))
        # i+=1
        # if i%10==0:
        #     print(id_to_data[int(id)])
id_to_data=np.array(list(id_to_data.values()))
fi = open("./id_to_data", "wb+")
pickle.dump(id_to_data, fi, protocol=4)
# json.dump(id_to_data, fp=fi)
# image = np.array2string(image)
# fi.write(image)
# print(i)
fi.close()
print("DOne1")
# # id_to_data=np.array(list(id_to_data.values()))
# # id_to_size=np.array(list(id_to_size.values()))
# f=open("./id_to_data","wb+")
# pickle.dump(id_to_data,f,protocol=4)
# f.close()
f=open("./id_to_size","wb+")
pickle.dump(id_to_size,f,protocol=4)
# id_to_size_file = np.array2string(id_to_size)
# f.write(id_to_size)
f.close()
id_to_box={}
print("DOne2")
with open("./bounding_boxes.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,box=line.split(" ",1)
        box=np.array([float(i) for i in box.split(" ")],dtype=np.float32)
        box[0]=box[0]/id_to_size[int(id)][1]*220
        box[1]=box[1]/id_to_size[int(id)][0]*220
        box[2]=box[2]/id_to_size[int(id)][1]*220
        box[3]=box[3]/id_to_size[int(id)][0]*220
        id_to_box[int(id)]=box
print("DOne3")
id_to_box=np.array(list(id_to_box.values()))
f=open("./id_to_box","wb+")
pickle.dump(id_to_box,f,protocol=4)
# id_to_box = np.array2string(id_to_box)
# f.write(id_to_box)
f.close()
print("DOne4")


id_to_data={}
id_to_size={}
i=0
open('./id_to_data_test', 'w+').close()
open('./id_to_size_test', 'w+').close()
# open('./id_to_box', 'w').close()
with open("./images_test.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,path=line.split(" ",1)
        image=Image.open("./images/"+path).convert('RGB')
        id_to_size[int(id)]=np.array(image,dtype=np.float32).shape[0:2]
        # print(np.size(image, 0))
        # print(np.size(image, 1))
        # print(np.size(image, 2))
        image=image.resize((110,110))
        image=np.array(image,dtype=np.float32)
        image=image/255
        image=Normalize(image,[0.485,0.456,0.406],[0.229,0.100,0.225])
        id_to_data[int(id)]=image
        print(id)
        # if int(int(id)/11788) == int(id)/11788:
        # print(int(int(id)/11788))
        # fi = open("./id_to_data", "a+")
        # fi.write(json.dumps(image.tolist()))

        # pickle.dump(image, fi, protocol=4)
        # image = np.array2string(image)
        # # fi.write(image)
        # # print(i)
        # fi.close()
        # print(np.size(image,0))
        # print(np.size(image, 1))
        # i+=1
        # if i%10==0:
        #     print(id_to_data[int(id)])
id_to_data=np.array(list(id_to_data.values()))
fi = open("./id_to_data_test", "wb+")
pickle.dump(id_to_data, fi, protocol=4)
# json.dump(id_to_data, fp=fi)
# image = np.array2string(image)
# fi.write(image)
# print(i)
fi.close()
print("DOne1")
# # id_to_data=np.array(list(id_to_data.values()))
# # id_to_size=np.array(list(id_to_size.values()))
# f=open("./id_to_data","wb+")
# pickle.dump(id_to_data,f,protocol=4)
# f.close()
f=open("./id_to_size_test","wb+")
pickle.dump(id_to_size,f,protocol=4)
# id_to_size_file = np.array2string(id_to_size)
# f.write(id_to_size)
f.close()
# id_to_box={}
# print("DOne2")
# with open("./bounding_boxes.txt") as f:
#     lines=f.read().splitlines()
#     for line in lines:
#         id,box=line.split(" ",1)
#         box=np.array([float(i) for i in box.split(" ")],dtype=np.float32)
#         box[0]=box[0]/id_to_size[int(id)][1]*110
#         box[1]=box[1]/id_to_size[int(id)][0]*110
#         box[2]=box[2]/id_to_size[int(id)][1]*110
#         box[3]=box[3]/id_to_size[int(id)][0]*110
#         id_to_box[int(id)]=box
# print("DOne3")
# id_to_box=np.array(list(id_to_box.values()))
# f=open("./id_to_box","wb+")
# pickle.dump(id_to_box,f,protocol=4)
# id_to_box = np.array2string(id_to_box)
# f.write(id_to_box)
# f.close()
print("DOne4")



import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, MaxPooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as tf
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from utils import getdata ,plot_model
from keras.callbacks import CSVLogger

import numpy as np


data_train,box_train,data_test,box_test=getdata()

# metric function
def my_metric(labels,predictions):
    threshhold=0.75
    x=predictions[:,0]*110
    x=tf.maximum(tf.minimum(x,110.0),0.0)
    y=predictions[:,1]*110
    y=tf.maximum(tf.minimum(y,110.0),0.0)
    width=predictions[:,2]*110
    width=tf.maximum(tf.minimum(width,110.0),0.0)
    height=predictions[:,3]*110
    height=tf.maximum(tf.minimum(height,110.0),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.tf.multiply(width,height)
    a2=tf.tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.tf.reduce_mean(sum)

# loss function
def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i]*110)
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.tf.where(condition,small_res,large_res)
    return tf.tf.reduce_mean(loss)


def resnet_block(inputs,num_filters,kernel_size,strides,activation='relu'):
    x=Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(inputs)
    x=BatchNormalization()(x)
    if(activation):
        x=Activation('relu')(x)
    return x


def resnet18():
    inputs=Input((110,110,3))
    
    # conv1
    x=resnet_block(inputs,64,[7,7],2)

    # conv2
    x=MaxPooling2D([3,3],2,'same')(x)
    for i in range(2):
        a=resnet_block(x,64,[3,3],1)
        b=resnet_block(a,64,[3,3],1,activation=None)
        x=keras.layers.add([x,b])
        x=Activation('relu')(x)
    
    # conv3
    a=resnet_block(x,128,[1,1],2)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=Conv2D(128,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,128,[3,3],1)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    # conv4
    a=resnet_block(x,256,[1,1],2)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=Conv2D(256,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,256,[3,3],1)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    # conv5
    a=resnet_block(x,512,[1,1],2)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=Conv2D(512,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,512,[3,3],1)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    x=AveragePooling2D(pool_size=4,data_format="channels_last")(x)
    # out:1*1*512

    y=Flatten()(x)
    # out:512
    y=Dense(1000,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)
    outputs=Dense(4,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)
    
    model=Model(inputs=inputs,outputs=outputs)
    return model

model = resnet18()


model.compile(loss=smooth_l1_loss, optimizer=Adam(), metrics=[my_metric])

model.summary()

def lr_sch(epoch):
    #200 total
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5

lr_scheduler=LearningRateScheduler(lr_sch)
lr_reducer=ReduceLROnPlateau(monitor='val_my_metric',factor=0.2,patience=5,mode='max',min_lr=1e-3)

checkpoint=ModelCheckpoint('model_new.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

csv_logger = CSVLogger('training.log', separator=',', append=False)
model_details=model.fit(data_train,box_train,batch_size=128,epochs=75,shuffle=True,validation_split=0.1,callbacks=[lr_scheduler,lr_reducer,checkpoint,csv_logger],verbose=1)

model.save('model_new_75.h5')

scores=model.evaluate(data_test,box_test,verbose=1)
print('Test loss : ',scores[0])
print('Test accuracy : ',scores[1])

plot_model(model_details)




import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import random
from keras import backend as tf
from utils import getdata
# data_train,box_train,data_test,box_test=getdata()


def my_metric(labels,predictions):
    threshhold=0.75
    x=predictions[:,0]*110
    x=tf.maximum(tf.minimum(x,110.0),0.0)
    y=predictions[:,1]*110
    y=tf.maximum(tf.minimum(y,110.0),0.0)
    width=predictions[:,2]*110
    width=tf.maximum(tf.minimum(width,110.0),0.0)
    height=predictions[:,3]*110
    height=tf.maximum(tf.minimum(height,110.0),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.tf.multiply(width,height)
    a2=tf.tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.tf.reduce_mean(sum)



def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i]*110)
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.tf.where(condition,small_res,large_res)
    return tf.tf.reduce_mean(loss)


plt.switch_backend('agg')

f=open("./data_test","rb+")
data=pickle.load(f)

f=open("./data_level","rb+")
box=pickle.load(f)

f=open("./id_to_size","rb+") 
size=pickle.load(f)
lenn = len(data)
index=[i for i in range(lenn)]
index=random.sample(index,100)


model=keras.models.load_model('./model_new.h5', custom_objects={'smooth_l1_loss': smooth_l1_loss, 'my_metric':my_metric})
result=model.predict(data[index,:,:,:])

mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
j=0
for i in index:
    print("Predicting "+str(i)+"th image.")
    true_box=box[i]
    image=data[i]
    prediction=result[j]
    j+=1
    for channel in range(3):
        image[:,:,channel]=image[:,:,channel]*std[channel]+mean[channel]

    image=image*255
    image=image.astype(np.uint8)
    plt.imshow(image)


    plt.gca().add_patch(plt.Rectangle((true_box[0],true_box[1]),true_box[2],true_box[3],fill=False,edgecolor='red',linewidth=2,alpha=0.5))
    plt.gca().add_patch(plt.Rectangle((prediction[0]*110,prediction[1]*110),prediction[2]*110,prediction[3]*110,fill=False,edgecolor='green',linewidth=2,alpha=0.5))
    plt.show()
    plt.savefig("./pred/"+str(i)+".png")
    plt.cla()






