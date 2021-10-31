import numpy as np
from utils import plotPredictions, plotTrainData, plotLoss, dice_coef, dice_coef_loss
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from model import unet

model_name = 'unet'
epochs = 7777
batch_size = 64
seed = 66

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.8)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='_save/best_' + model_name + '.hdf5')

Adam = Adam(learning_rate= 3e-4) 
SGD = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# 1.data
x = np.load('../_data/_npy/img.npy')
y = np.load('../_data/_npy/mask.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                        train_size=0.8, random_state=seed, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                        train_size=0.8, random_state=seed, shuffle=True)

# plt.imshow(x_train[3:4])
# plt.show()

plotTrainData(x_train, y_train, 'train')
plotTrainData(x_test, y_test, 'test')
plotTrainData(x_val, y_val, 'valid')

# 2. model
model = unet()
model.summary()
plot_model(model, to_file='_save/' + model_name +'.png', show_shapes=True)

# 3. compile, fit
model.compile(loss=dice_coef_loss, optimizer=SGD, metrics=[dice_coef])

hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                validation_data=(x_val, y_val), verbose=1, callbacks=[lr, es, cp])

model.save('_save/seg_' + model_name + '.h5')

# 4. evaluate, predict

np.random.seed(seed)
tf.random.set_seed(seed)
plotLoss(hist)
plotPredictions(x_train, y_train, x_val, y_val, x_test, y_test, model)