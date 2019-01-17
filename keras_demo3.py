from keras.datasets import reuters
import numpy as np
from keras import Sequential
from keras.layers import Activation, Dense
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(train_data, train_lables), (test_data, test_lables) = reuters.load_data(num_words=10000)

"""
word_index 将单词映射为正式索引字典
i-3  0,1,2位是padding填充
"""
word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# 编码数据
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results
#
#
# one_hot_train_labels = to_one_hot(train_lables)
# one_hot_test_labels = to_one_hot(test_lables)

"""
等价于to_categorical 
"""
one_hot_train_labels = to_categorical(train_lables)
one_hot_test_labels = to_categorical(test_lables)


"""
构建网络
"""
# # 模型定义
# model = Sequential([
#     Dense(64, input_dim=10000),
#     Activation('relu'),
#     Dense(64),
#     Activation('relu'),
#     Dense(46),
#     Activation('softmax')
# ])
#
# # 模型编译
# model.compile(optimizer='rmsprop',
#               loss="categorical_crossentropy",
#               metrics=['accuracy']
#               )
#
# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
#
# # 训练模型
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(loss) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.clf()  # 清空图像
# acc = history.history['acc']
# val_loss = history.history['val_acc']
#
# plt.plot(epochs, acc, 'bo', label='Train acc')
# plt.plot(epochs, val_loss, 'b', label='Validation acc')
# plt.title('Train and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()

# 九轮过后，开始过拟合，从新训练新网络，共9次迭代
"""
构建网络2
"""
model = Sequential([
    Dense(64, input_dim=10000),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(46),
    Activation('softmax')
])

# 模型编译
model.compile(optimizer='rmsprop',
              loss="categorical_crossentropy",
              metrics=['accuracy']
              )

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)
print(results)


# 在新数据上生成预测结果
predictions = model.predict(x_test)
print(predictions[0].shape)
print(np.sum(predictions[0]))
# 最大预测类别
print(np.argmax(predictions[0]))
