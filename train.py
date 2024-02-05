import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

MODEL_PATH = "./models/ann3.h5"
DATA_PATH = "./data/emotion_landmarks.csv"
HISTORY_DIR = "./tmp"

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.00005
TEST_SPLIT = 0.2


def run(
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    learning_rate=LEARNING_RATE,
    test_split=TEST_SPLIT,
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    history_dir=HISTORY_DIR
):
    OPT = keras.optimizers.Adam(learning_rate = learning_rate)

    df = pd.read_csv(data_path)

    print("dataframe shape:", df.shape)

    df2 = df.groupby(['emotion_label'])['emotion_label'].count()
    print(df2)

    X = df.iloc[:,2:].values
    y = df.iloc[:,1].values

    ohe = OneHotEncoder()
    y = ohe.fit_transform(y.reshape(-1, 1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_split,shuffle=True, random_state=0)

    print("X_train shape:",X_train.shape)
    print("X_test shape:",X_test.shape)
    print("y_train shape:",y_train.shape)
    print("y_test shape:",y_test.shape)

    model = Sequential()
    model.add(Dense(2048, input_dim=X.shape[1], activation="relu")) # 5778*2048+2048*1024+1024*512+512*512+512*7
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(y.shape[1], activation="softmax"))

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=epochs, batch_size=batch_size)

    model.save(model_path)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(history_dir,'training_accuracy_history.png'))
    # plt.show(block=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(history_dir,'training_loss_history.png'))
    # plt.show(block=True)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS, help='number of epochs or iteration for training')
    parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE, help='number of rows of data in each batch')
    parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE, help='learning rate for adam algorithm')
    parser.add_argument('-ts', '--test-split', type=int, default=TEST_SPLIT, help='split percentage of the test set')
    parser.add_argument('-dp', '--data-path', type=str, default=DATA_PATH, help='path of the dataset file (.csv)')
    parser.add_argument('-mp', '--model-path', type=str, default=MODEL_PATH, help='resulting model path and name after training')
    parser.add_argument('-hd', '--history-dir', type=str, default=HISTORY_DIR, help='directory of the training history graphs (accuracy/loss)')

    opt = parser.parse_args()
    return opt


def main(opt):
    print(opt)
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)