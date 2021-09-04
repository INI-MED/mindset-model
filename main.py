import ast
import codecs
import csv

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

arr_train = []
arr_test = []
new_arr_train_10 = []
index = 0
y_train = []
x_train = []
f = codecs.open('db.txt', 'r', "utf_8_sig")
ff = f.read()
items = ast.literal_eval(ff)


def tex_to_int(st):
    text = ":".join("{:02x}".format(ord(c)) for c in st)
    return ''.join(str(int(h, 16)) for h in text.split(':'))


def correct_data(row):
    if row[7] == 'Москва':
        row[7] = '1'
    elif row[7] == 'Санкт-Петербург':
        row[7] = '2'

    if row[16] == 'M':
        row[16] = '0'
    elif row[16] == 'F':
        row[16] = '1'
    row[10] = items[0][row[10]]
    row[11] = items[1][row[11]]
    row[28] = items[2][row[28]]
    if row[29] == "N":
        row[29] = '-1'
    if row[26] == "N":
        row[26] = '-1'
    if row[15] == 'N':
        row[15] = '-1'

    row[17] = tex_to_int(row[17])
    row[18] = tex_to_int(row[18])
    row[19] = tex_to_int(row[19])
    row[20] = tex_to_int(row[20])
    return row


with open("data.csv", encoding='utf-8') as r_file:
    file_reader = csv.reader(r_file, delimiter=";")
    for row in file_reader:
        row = correct_data(row)

        if row[0] == "TRAIN":
            arr_train.append(row)
        else:
            arr_test.append(row)

for i in arr_train:
    y_train.append(i[4])
    x_train.append([i[1], i[2], i[3], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14], i[15], i[16],
                    i[17], i[18], i[19], i[20], i[21], i[22], i[23], i[24], i[25], i[26], i[27], i[28], i[29]])

y_train = np.array(y_train, float)
x_train = np.array(x_train, float)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
model.save('16_model_2.h5')
model = keras.models.load_model('16_model_2.h5')

test = ['TEST', '13', '1', '1', '0', '54', '6', 'Москва', '30', '8', 'Volkswagen', 'Tiguan', '170', '0', '1063335',
        '352', 'F', '1S', '1S', '0', '0', '1', '1', '0', '2', '0', '1', '0', 'Москва', '-0.29']
test2 = correct_data(test)
x_test = np.array([[i[1], i[2], i[3], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14], i[15], i[16],
                    i[17], i[18], i[19], i[20], i[21], i[22], i[23], i[24], i[25], i[26], i[27], i[28], i[29]]],
                  float)
x_test -= mean
x_test /= std
