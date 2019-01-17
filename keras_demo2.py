from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

(train_data, train_lables), (test_data, test_lables) = imdb.load_data(num_words=10000)  # 保留最常出现的10,000个单词
# print(train_data[2])
# print(train_lables[0])
# print(max([max(sequence) for sequence in train_data]))  # 限定10000个常见，单词索引不会超过10,000

"""
word_index 将单词映射为正式索引字典
i-3  0,1,2位是padding填充
"""
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])


def vectorize_sequences(sequences, dimension=10000):
    """
    句子向量化标表示
    enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)
    组合为一个索引序列，同时列出数据和数据下标
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # 将results[i]的指定索引设置为1
    return results


# 将训练/测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# print(x_test[0])
# print(x_train[0][6])


# 标签向量化
y_train = np.array(train_lables).astype('float32')
y_test = np.array(test_lables).astype("float32")
# print(y_train.shape)


# 模型定义
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_dim=10000))
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# # # 编译模型
# # model.compile(optimizer='rmsprop',
# #               loss='binary_crossentropy',
# #               metrics=['accuracy'])
# #
# # # 配置优化器
# # model.compile(optimizer=optimizers.RMSprop(lr=0.001),
# #               loss='binary_crossentropy',
# #               metrics=['accuracy'])
#
#
# # 留出验证集
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
#
# # 模型训练
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))
# results = model.evaluate(x_val, y_val)
# print(results)
#
#
# # 绘制训练损失和验证损失
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
#
# epochs = range(1, len(loss_values) + 1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# # 绘制训练精度和验证及精度
# plt.clf()
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']
#
# plt.plot(epochs, acc_values, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()


# 重新训练模型
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
predict = model.predict(x_test)
print(predict)