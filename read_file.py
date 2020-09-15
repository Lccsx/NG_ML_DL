import tensorflow as tf
import numpy as np
import struct
import matplotlib.pyplot as plt
def read_image(file_name):

    '''
    :param file_name: 文件路径
    :return:  训练或者测试数据
    如下是训练的图片的二进制格式
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    '''

    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中
    head = struct.unpack_from('>IIII', file_content, 0)  # 取前4个整数，返回一个元组
    offset = struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    width = head[2]   #宽度
    height = head[3]  #高度
    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
    imgs = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    imgs_array=np.array(imgs).reshape((imgNum,width*height))     #最后将读取的数据reshape成 【图片数，图片像素】二维数组
    return imgs_array
def out_image(img):
    '''
    :param img: 图片像素组
    :return:
    '''
    plt.figure()
    plt.imshow(img)
    plt.show()
def read_label(file_name):
    '''
    :param file_name:
    :return:
    标签的格式如下：
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    '''
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')
    labelNum = head[1]  # label数
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)
def get_data():
    # 文件获取
    train_image = "C:/Users/东邪/PycharmProjects/NG_ML_DL/2020/dataset/fashin_mnist/train-images-idx3-ubyte/train-images-idx3-ubyte"
    test_image = "C:/Users/东邪/PycharmProjects/NG_ML_DL/2020/dataset/fashin_mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    train_label = "C:/Users/东邪/PycharmProjects/NG_ML_DL/2020/dataset/fashin_mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    test_label = "C:/Users/东邪/PycharmProjects/NG_ML_DL/2020/dataset/fashin_mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
    # 读取数据
    training_images = read_image(train_image)
    test_images = read_image(test_image)
    training_labels = read_label(train_label)
    test_labels = read_label(test_label)
    print(training_labels[0:10])
    print(test_labels[0:10])
    out_image(np.array(training_images[24]).reshape(28, 28))
    return training_images, training_labels, test_images, test_labels

if __name__ == "__main__":
    (training_images, training_labels), (test_images, test_labels) = get_data()


    #plt.imshow(training_images[])
    print(training_labels[0])
    print(training_images[0])

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(128, activation=tf.nn.relu)
                                tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.evaluate(test_images, test_labels)
    model.fit(training_images, training_labels, epochs=5)
