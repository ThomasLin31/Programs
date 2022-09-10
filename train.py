# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout

# 載入mnist資料集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape正規化
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 轉成浮點數(為了正規化)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 將灰階(2**8-1=255)的圖片正規化
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('x_train數量', x_train.shape[0])
print('x_test數量', x_test.shape[0])


# 創建序列模型，並加上layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax)) # 數字 0~9
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

# 可調整epochs
model.fit(x=x_train,y=y_train, epochs=10)
model.evaluate(x_test, y_test)


model.save('newtest.h5')  # 儲存model
del model 