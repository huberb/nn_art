from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.optimizers import Adam
from keras.initializers import RandomUniform, RandomNormal
import matplotlib.pyplot as plt
import numpy as np
import os.path

def create_model():
    initializer = RandomNormal(mean=0, stddev=1)
    model = Sequential()
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer, input_shape=(2,)))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(32, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(3, activation='sigmoid', kernel_initializer=initializer))
    model.compile(Adam(), loss='mean_squared_error')
    # model.summary()
    return model

def create(width, height):
    print("init..")

    img = [list(range(0, width)) for _ in list(range(0, height))]

    model = create_model()

    print("creating image..")
    for x in range(0, width):
        for y in range(0, height):
            test = [(x/width) - 0.5, (y/height) - 0.5]
            prediction = model.predict(np.array([test]))
            img[x][y] = prediction[0]
    print("finished!")

    print("saving image..")
    plt.imshow(img)
    plt.show()
    # plt.savefig(path)

if __name__ == "__main__":
    create(256, 256)
