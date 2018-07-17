from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np
import keras.backend as K
K.set_image_data_format('channels_last')

# 激活函数
def swish(x):
    return (K.sigmoid(x) * x)

# 更新定制对象
get_custom_objects().update({'swish': swish})

# 定义手写字符模型
def MnistModel(input_shape):

	# 输入
	X_input = Input(input_shape)

	# 用零填充
	X = ZeroPadding2D((3, 3))(X_input)

	# 卷积神经网络
	X = Conv2D(32, (5, 5), strides=(1, 1), name='conv0', padding='same')(X) 	# 卷积层
	X = Activation('swish')(X)													# 激活函数
	X = Conv2D(32, (5, 5), strides=(1, 1), name='conv1', padding='same')(X)	# 卷积层
	X = Activation('swish')(X)													# 激活函数

	X = MaxPooling2D((2, 2), name='max_pool0')(X)								# 池化
	X = Dropout(0.25)(X)														# dropout

	X = Conv2D(64, (3, 3), strides=(1, 1), name='conv2', padding='same')(X) 	# 卷积层
	X = Activation('swish')(X)													# 激活函数
	X = Conv2D(64, (3, 3), strides=(1, 1), name='conv3', padding='same')(X) 	# 卷积层
	X = Activation('swish')(X)													# 激活函数

	X = MaxPooling2D((2, 2), name='max_pool2')(X)								# 池化
	X = Dropout(0.25)(X)														# dropout

	X = Flatten()(X)															# 将多维的输入一维化
	X = Dense(1024, activation='swish', name='fc0')(X)							# 全连接层
	X = Dropout(0.5)(X)															# dropout
	X = Dense(10, activation='sigmoid', name='fc1')(X)							# 全连接层

	# 建立模型
	model = Model(inputs=X_input, outputs=X, name='MnistModel')

	return model

# 载入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据从[0,255]转成[0,1]，处理速度更快
X_train = X_train / 255.0
X_test = X_test / 255.0

# 将1维向量中的784个值转换成28x28x1的3维矩阵
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# 对label数据进行转化，one—hot编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 打印数据集的大小
print('训练集特征大小', X_train.shape)
print('训练集标签大小', y_train.shape)
print('测试集特征大小', X_test.shape)
print('测试集标签大小', y_test.shape)

# 数据集划分
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)

# 建立模型
mnistModel = MnistModel(X_train.shape[1:])
optimizer = adam(lr=0.001, epsilon=1e-08, decay=0.0)		# 选用Adam优化器
mnistModel.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 退火算法
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001)

epochs = 30													# 迭代次数
batch_size = 86												# batch大小

# 数据增强，防止过拟合
datagen = ImageDataGenerator(
        featurewise_center=False,  				# set input mean to 0 over the dataset
        samplewise_center=False,  				# set each sample mean to 0
        featurewise_std_normalization=False,  	# divide inputs by std of the dataset
        samplewise_std_normalization=False,  	# divide each input by its std
        zca_whitening=False,  					# apply ZCA whitening
        rotation_range=10,  					# randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, 						# Randomly zoom image
        width_shift_range=0.1,  				# randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  				# randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  				# randomly flip images
        vertical_flip=False)  					# randomly flip images

# 训练模型
datagen.fit(X_train)
history = mnistModel.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size,
							  callbacks=[learning_rate_reduction,TensorBoard(log_dir='./log')])

# 保存训练好的模型
mnistModel.save('./CNN_Mnist.h5')
print('模型保存成功!')

# 对训练好的模型进行评估
score = mnistModel.evaluate(X_test, y_test, verbose=0)
print(score)

# 测试集
predicts = mnistModel.predict(X_test)
predicts = np.argmax(predicts,axis = 1)
expects = np.argmax(y_test,axis = 1)

count = 0
num = 0
for p,e in zip(predicts,expects):
	num += 1
	if p == e:
		count += 1

print('\n----------------------------------')
print('测试集数量： ',num)
print('正确的数量： ',count)
print('测试准确率： ',float(count)/float(num))
print('\n----------------------------------')