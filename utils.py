import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K

def plotTrainData(a,b,c):
    for i in range(3):
        ix = np.random.randint(0, len(a))
        plt.subplot(1,3,1)
        plt.title("X_" + c)
        plt.imshow(a[ix])
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.title("y_" + c)
        plt.imshow(b[ix])
        plt.axis('off')

        masked_image = a[ix] * 0.5 + b[ix] * 0.5
        plt.subplot(1,3,3)
        plt.title('Masked Image')
        plt.imshow(masked_image)
        plt.axis('off')
        plt.show()

def plotLoss(hist):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['dice_coef'], 'b', label='train dice_coef')
    acc_ax.plot(hist.history['val_dice_coef'], 'g', label='val dice_coef')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('dice_coef')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

def plotPredictions(x_train, y_train, x_val, y_val, X_test, y_test, model):
    # train
    ix = np.random.randint(0, len(x_train))
    input_ = x_train[ix:ix+1]
    mask = y_train[ix:ix+1]
    preds_train = model.predict(input_)
    # preds_train_t = (preds_train > 0.5).astype(np.uint8)
    plt.figure(figsize=(10,10))
    plt.subplot(1,4,1)
    plt.title("x_train")
    plt.axis('off')
    plt.imshow(input_[0])
    plt.subplot(1,4,2)
    plt.title("y_train")
    plt.axis('off')
    plt.imshow(mask[0])
    plt.subplot(1,4,3)
    ret = model.evaluate(input_, mask)
    plt.title("Prediction: %.4f" % (ret[1]))
    plt.axis('off')
    plt.imshow(preds_train[0])
    masked_image = input_[0] * 0.5 + preds_train[0] * 0.5
    plt.subplot(1,4,4)
    ret = model.evaluate(input_, mask)
    plt.title("Masked pred_Image")
    plt.axis('off')
    plt.imshow(masked_image)
    plt.show()

    # valid
    ix = np.random.randint(0, len(x_val))
    input_ = x_val[ix:ix+1]
    mask = y_val[ix:ix+1]
    preds_valid = model.predict(input_)
    # preds_valid_t = (preds_valid > 0.5).astype(np.uint8)
    plt.figure(figsize=(10,10))
    plt.subplot(1,4,1)
    plt.title("x_valid")
    plt.axis('off')
    plt.imshow(input_[0])
    plt.subplot(1,4,2)
    plt.title("y_valid")
    plt.axis('off')
    plt.imshow(mask[0])
    plt.subplot(1,4,3)
    ret = model.evaluate(input_, mask)
    plt.title("Prediction: %.4f" % (ret[1]))
    plt.axis('off')
    plt.imshow(preds_valid[0])
    masked_image = input_[0] * 0.5 + preds_valid[0] * 0.5
    plt.subplot(1,4,4)
    ret = model.evaluate(input_, mask)
    plt.title("Masked pred_Image")
    plt.axis('off')
    plt.imshow(masked_image)
    plt.show()

    #test
    ix = np.random.randint(0, len(X_test))
    input_ = X_test[ix:ix+1]
    mask = y_test[ix:ix+1]
    preds_test = model.predict(input_)
    # preds_test_t = (preds_test > 0.5).astype(np.uint8)
    plt.figure(figsize=(10,10))
    plt.subplot(1,4,1)
    plt.title("x_test")
    plt.axis('off')
    plt.imshow(input_[0])
    plt.subplot(1,4,2)
    plt.title("y_test")
    plt.axis('off')
    plt.imshow(mask[0])
    plt.subplot(1,4,3)
    ret = model.evaluate(input_, mask)
    plt.title("Prediction: %.4f" % (ret[1]))
    plt.axis('off')
    plt.imshow(preds_test[0])
    masked_image = input_[0] * 0.5 + preds_test[0] * 0.5
    plt.subplot(1,4,4)
    ret = model.evaluate(input_, mask)
    plt.title("Masked pred_Image")
    plt.axis('off')
    plt.imshow(masked_image)
    plt.show()

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)