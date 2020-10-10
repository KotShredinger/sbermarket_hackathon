# -*- coding: utf-8 -*-

# uncomment in Google Colab
#!pip install gdown
#!gdown https://drive.google.com/uc?id=1wdIU5CSL4Ug4-JSnI4J5PA980tLYTSk_

from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd

def convertToDatetime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

def train_generator():
    while True:
        x_train = np.random.random((1000, sequence_length, 4))
        # y_train will depend on past 5 timesteps of x
        y_train = x_train[:, :, 0]
        for i in range(1, 5):
            y_train[:, i:] += x_train[:, :-i, i]
        # y_train = to_categorical(y_train > 2.5)
        yield x_train, y_train

orders = pd.read_csv('tab_1_orders.csv')

user2time = orders.groupby('user_id')['order_created_time'].apply(list)

vals = user2time.values

N = 5 # batch size (how many points we observed before)

vals_n = [el for el in vals if len(el) >= N]

dt_diff = datetime(2019, 12, 31, 21, 29, 17)

vals_dt = []
for l in vals_n:
  nl = []
  for el in l:
    dt_ = convertToDatetime(el)
    nl.append(dt_)
  vals_dt.append(nl)

for i in range(len(vals_dt)):
  for j in range(len(vals_dt[i])):
    vals_dt[i][j] = (vals_dt[i][j] - dt_diff).total_seconds()

X, y = [], []
for l in vals_dt:
  for i in range(len(l) - N + 1):
    x_ = l[i:i+N-1]
    y_ = l[i+N-1]
    X.append(x_)
    y.append(y_)

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0],X.shape[1],1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(N-1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=10, validation_split=0.2, verbose=1)

model.save("timeLstm")

