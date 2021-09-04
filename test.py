import ast
import codecs

import numpy as np
from tensorflow import keras

f = codecs.open('db.txt', 'r', "utf_8_sig")
ff = f.read()
items = ast.literal_eval(ff)


def tex_to_int(st):
    text = ":".join("{:02x}".format(ord(c)) for c in st)
    return ''.join(str(int(h, 16)) for h in text.split(':'))


def correct_data(row):
    if row[5] == 'Москва':
        row[5] = '1'
    elif row[5] == 'Санкт-Петербург':
        row[5] = '2'

    if row[14] == 'M':
        row[14] = '0'
    elif row[14] == 'F':
        row[14] = '1'

    row[8] = items[0][row[8]]
    row[9] = items[1][row[9]]
    row[26] = items[2][row[26]]
    if row[24] == "N":
        row[24] = '-1'
    if row[13] == 'N':
        row[13] = '-1'

    row[15] = tex_to_int(row[15])
    row[16] = tex_to_int(row[16])
    row[17] = tex_to_int(row[17])
    row[18] = tex_to_int(row[18])
    return row


test = ['13', '1', '1', '54', '6', 'Москва', '30', '8', 'Volkswagen', 'Tiguan', '170', '0',
        '1063335', '352', 'F', '1S',
        '1S', '0', '0', '1', '1', '0', '2', '0', '1', '0', 'Москва', '-0.29', '2', '2', '1']
x = correct_data(test)
x = np.array(x, float)

mean = x.mean(axis=0)
std = x.std(axis=0)
x -= mean
x /= std

x_test = np.array([x], float)
model = keras.models.load_model('16_model_2.h5')

pred = model.predict(x_test)
print(pred)
