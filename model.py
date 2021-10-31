from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import *
from tensorflow.python.keras.backend import relu


def unet(input_size=(256,256,3)):
    inputs = Input(input_size)
    
    c1 = Conv2D(32, 3, padding='same')(inputs)
    c1 = ReLU()(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, 3, padding='same')(c1)
    c1 = ReLU()(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPool2D((2, 2))(c1)

    c2 = Conv2D(64, 3, padding='same')(p1)
    c2 = ReLU()(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, 3, padding='same')(c2)
    c2 = ReLU()(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPool2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(128, 3, padding='same')(p2)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, 3, padding='same')(c3)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPool2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(256, 3, padding='same')(p3)
    c4 = ReLU()(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, 3, padding='same')(c4)
    c4 = ReLU()(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPool2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(512, 3, padding='same')(p4)
    c5 = ReLU()(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(512, 3, padding='same')(c5)
    c5 = ReLU()(c5)
    c5 = BatchNormalization()(c5)

    u1 = UpSampling2D((2,2))(c5)
    concat1 = concatenate([c4, u1])

    c6 = Conv2D(256, 3, padding='same')(concat1)
    c6 = ReLU()(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256,3, padding='same')(c6)
    c6 = ReLU()(c6)
    c6 = BatchNormalization()(c6)

    u2 = UpSampling2D((2,2))(c6)
    concat2 = concatenate([c3, u2])

    c7 = Conv2D(128, 3, padding='same')(concat2)
    c7 = ReLU()(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, 3, padding='same')(c7)
    c7 = ReLU()(c7)
    c7 = BatchNormalization()(c7)

    u3 = UpSampling2D((2,2))(c7)
    concat3 = concatenate([c2, u3])
    c8 = Conv2D(64, 3, padding='same')(concat3)
    c8 = ReLU()(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, 3, padding='same')(c8)
    c8 = ReLU()(c8)
    c8 = BatchNormalization()(c8)
    
    u4 = UpSampling2D((2,2))(c8)
    concat4 = concatenate([c1, u4])

    c9 = Conv2D(32, 3, padding='same')(concat4)
    c9 = ReLU()(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(32, 3, padding='same')(c9)
    c9 = ReLU()(c9)
    c9 = BatchNormalization()(c9)

    c10 = Conv2D(3, 1, activation='sigmoid',  padding='same')(c9)


    model = Model(inputs=[inputs], outputs=[c10])

    return model