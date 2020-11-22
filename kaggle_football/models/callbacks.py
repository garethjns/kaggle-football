from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard


class Callbacks:
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=2, patience=20)
    tb = TensorBoard(histogram_freq=5)
