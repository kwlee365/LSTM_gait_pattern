# In[]

def initialize_lstm_model(X):

    _DROPOUT_PERCENT = 0.2

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.layers import LSTM

    model = Sequential()  # Define the model to be a Sequential one

    model.add(LSTM(units=32,
                   activation="tanh",
                   return_sequences=True,
                   input_shape=(X.shape[1], X.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    model.add(LSTM(units=64,
                   activation="tanh",
                   return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    model.add(LSTM(units=64,
                   activation="tanh",
                   return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(_DROPOUT_PERCENT))

    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(100, activation="softmax"))

    return model


def compile_lstm_model(
        model,
        optimization_option="adam",
        loss_function="CategoricalCrossentropy",
        metrics=["accuracy"]
):
    # This function set the optimization parameters to compile the LSTM-Model
    from tensorflow.keras.optimizers import Adam
    if optimization_option == "adam":
        optimization_option = Adam(learning_rate=1e-3)

    model.compile(optimizer=optimization_option,
                  loss=loss_function, metrics=metrics)

    return model


def set_monitor_lstm_model(verbose=1, monitor="loss", patience=100):
    # This function sets the monitors for the LSTM-Model
    from tensorflow.keras.callbacks import EarlyStopping

    monitor = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        mode="auto",
        restore_best_weights=True
    )
    return monitor


def fit_lstm_model(model, X_train, Y_train, X_test, Y_test, monitor, epochs=10):
    # This function fits the lastm model by using the given input data
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=[monitor]
    )
    return history
# %%
