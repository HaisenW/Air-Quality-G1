# NN training for both tasks
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from imblearn.over_sampling import SMOTE
import tensorflow as tf

RANDOM_STATE = 42

def nn_reg_model(X_train, y_train):
    # mlp_reg = Pipeline([('scaler', StandardScaler()),
    #                 ('rgs', MLPRegressor(hidden_layer_sizes=(16,),
    #                                      activation='logistic',
    #                                      random_state=RANDOM_STATE,
    #                                      learning_rate_init=0.001))])
    # mlp_reg.fit(X_train, y_train)
    #
    # return mlp_reg
    tf.random.set_seed(RANDOM_STATE)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.fit_transform(X_val)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(11,), name="input_layer"),

        tf.keras.layers.Dense(128, activation="relu", name="dense_layer", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, activation="relu", name="dense_layer2", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu', name="dense_layer3", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

        tf.keras.layers.Dense(1, name="output_layer")
    ])
    model.compile(loss=tf.keras.losses.mae,
                  optimizer="adam",
                  metrics=['mae'])
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )
    model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, callbacks=[early_stop], validation_data=(X_val_scaled, y_val))

    return model

def nn_class_model(X_train, y_train):
    # mlp_class = Pipeline([('scaler', StandardScaler()),
    #                         ('resampler', SMOTE(random_state=RANDOM_STATE)),
    #                         ('rgs', MLPClassifier(hidden_layer_sizes=(16,),
    #                         activation='logistic',
    #                         random_state=RANDOM_STATE,
    #                         learning_rate_init=0.001))])
    # mlp_class.fit(X_train, y_train)
    #
    # return mlp_class
    tf.random.set_seed(RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.fit_transform(X_val)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(11,), name="input_layer"),

        tf.keras.layers.Dense(128, activation="relu", name="dense_layer", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, activation="relu", name="dense_layer2", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation="relu", name="dense_layer3", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),

        tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer="adam",
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
                  )
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor = "val_loss"
    )
    model.fit(X_resampled, y_resampled, validation_data=(X_val_scaled, y_val), epochs=200, batch_size=32, callbacks=[early_stop])

    return model