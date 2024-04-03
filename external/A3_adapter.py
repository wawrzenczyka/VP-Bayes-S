import numpy as np
import tensorflow as tf

from .A3.libs.A3 import A3
from .A3.libs.architecture import (
    RandomNoise,
    VariationalAutoEncoder,
    alarm_net,
    dense_ae,
)


class A3Adapter:
    def __init__(self, target_epochs=30, a3_epochs=60):
        self.target_epochs = target_epochs
        self.a3_epochs = a3_epochs

    def fit(self, X_train, y_train=None):
        train_target = (X_train, X_train)
        val_target = (X_train, X_train)  # fake!

        train_alarm = (X_train, np.zeros(len(X_train)))
        val_alarm = (X_train, np.zeros(len(X_train)))  # fake!

        TRAIN_TARGET = True

        random_noise = RandomNoise("normal")
        model_vae = VariationalAutoEncoder(
            input_shape=X_train.shape[1:], layer_dims=[800, 400, 100, 25]
        )
        model_vae.compile(optimizer=tf.keras.optimizers.Adam(0.001))
        # Subclassed Keras models don't know about the shapes in advance... build() didn't do the trick
        model_vae.fit(train_target[0], epochs=0, batch_size=256)

        if TRAIN_TARGET:
            # Create target network
            model_target = dense_ae(
                input_shape=X_train.shape[1:], layer_dims=[1000, 500, 200, 75]
            )
            model_target.compile(optimizer="adam", loss="binary_crossentropy")
            model_target.fit(
                train_target[0],
                train_target[1],
                validation_data=val_target,
                epochs=self.target_epochs,
                batch_size=256,
            )

            # Create alarm and overall network
            model_a3 = A3(target_network=model_target)
            model_a3.add_anomaly_network(random_noise)
            model_alarm = alarm_net(
                layer_dims=[1000, 500, 200, 75],
                input_shape=model_a3.get_alarm_shape(),
            )
            model_a3.add_alarm_network(model_alarm)

        model_a3.compile(
            optimizer=tf.keras.optimizers.Adam(0.00001),
            loss="binary_crossentropy",
        )
        model_a3.fit(
            train_alarm[0],
            train_alarm[1],
            validation_data=val_alarm,
            epochs=self.a3_epochs,
            batch_size=256,
            verbose=1,
        )

        self.model = model_a3
        return self

    def score_samples(self, samples):
        return 1 - self.model.predict(samples).reshape(-1)
