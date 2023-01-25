import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from abc import abstractmethod


class ImageClassifierBase:

    def __init__(self, input_shape, lr, mc=True, metrics=True, trunc=False, trained_model=None, model_name="test"):
        self.input_shape = input_shape
        self.lr = lr
        self.mc = mc
        self.metrics = metrics
        self.trunc = trunc
        self.trained_model = trained_model
        self.model_name = model_name + "_with_mc" if self.mc else model_name + "_without_mc"

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        feature_extraction_output = self._feature_extraction(inputs)
        output = self._fusion_layer(*feature_extraction_output)
        output = self._classifier(output)

        model = Model(inputs=inputs, outputs=output)

        if self.trained_model:
            for i, layer in enumerate(model.layers):
                layer.set_weights(self.trained_model.layers[i].get_weights())

        callbacks = None if self.trunc else self._get_callbacks()

        return model, callbacks

    def _compile_model(self, model):
        adam = tf.keras.optimizers.Adam(lr=self.lr)

        compile_dict = {
            "optimizer": adam,
            "loss": "categorical_crossentropy"
        }

        if self.metrics:
            compile_dict["metrics"] = ['accuracy', self._get_metrics()]

        model.compile(**compile_dict)

        return model

    def _get_callbacks(self):
        model_checkpoint = ModelCheckpoint(f"{self.model_name}.h5", monitor='val_accuracy', mode='max', verbose=1,
                                           save_best_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, verbose=1, patience=5)
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=30)
        return [reduce_lr, es, model_checkpoint]

    @abstractmethod
    def _feature_extraction(self, inputs):
        pass

    def _fusion_layer(self, *args):
        flattened = [Flatten(layer) for layer in args]
        concatenated_tensor = Concatenate(axis=1)(flattened)
        return concatenated_tensor

    @abstractmethod
    def _classifier(self, concatenated_features):
        pass

    def _get_metrics(self):
        return [
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')]

    def _get_dropout(self, input_tensor, rate):
        if self.mc:
            return Dropout(rate=rate)(input_tensor, training=True)
        else:
            return Dropout(rate=rate)(input_tensor)


# Our Proposed Fusion Model:
class FusionModel(ImageClassifierBase):

    def __init__(self,  input_shape=(150, 150, 1), lr=0.00005, mc=True, metrics=True, trunc=False, trained_model=None, model_name="test"):
        super().__init__(input_shape, lr, mc, metrics, trunc, trained_model, model_name)

    def _feature_extraction(self, inputs):
        input2 = tf.stack([inputs, inputs, inputs], axis=3)[:, :, :, :, 0]
        vgg_model = tf.keras.applications.VGG16(weights='imagenet',
                                                include_top=False,
                                                input_shape=(self.input_shape[0], self.input_shape[1], 3))
        vgg_model.trainable = False

        vgg_feature = vgg_model(input2)
        # First conv block
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv1 = MaxPool2D(pool_size=(2, 2))(conv1)

        # Second conv block
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPool2D(pool_size=(2, 2))(conv2)

        # Third conv block
        conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = MaxPool2D(pool_size=(2, 2))(conv3)

        # Fourth conv block
        conv4 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
        conv4 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='target_layer')(
            conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = MaxPool2D(pool_size=(2, 2))(conv4)
        conv4 = self._get_dropout(conv4, rate=0.2)

        # Fifth conv block
        conv5 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
        conv5 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = MaxPool2D(pool_size=(2, 2))(conv5)
        conv4 = self._get_dropout(conv4, rate=0.2)

        output_list = [conv3, conv4, conv5, vgg_feature]

        return output_list

    def _classifier(self, concatenated_features):
        x = Flatten()(concatenated_features)
        x = Dense(units=512, activation='relu')(x)

        if not self.trunc:

            x = self._get_dropout(x, rate=0.7)
            x = Dense(units=128, activation='relu')(x)
            x = self._get_dropout(x, rate=0.5)
            x = Dense(units=64, activation='relu')(x)
            x = self._get_dropout(x, rate=0.3)
            x = Dense(3, activation='softmax')(x)

        return x


# Simple CNN Model:
class SimpleCNNModel(ImageClassifierBase):

    def __init__(self,  input_shape, lr, mc=True, metrics=True, trunc=False, trained_model=None, model_name="test"):
        super().__init__(input_shape, lr, mc, metrics, trunc, trained_model, model_name)

    def _feature_extraction(self, inputs):
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)

        conv2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv2 = MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2)

        conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = self._get_dropout(conv5, rate=0.2)

        return [conv5]

    def _classifier(self, concatenated_features):
        x = Flatten()(concatenated_features)

        x = Dense(units=128, activation='relu')(x)

        if not self.trunc:
            x = self._get_dropout(x, rate=0.7)
            x = Dense(units=64, activation='relu')(x)
            x = self._get_dropout(x, rate=0.5)

            x = Dense(3, activation='softmax')(x)

        return x


# Multi-headed Model:
class MultiHeadedModel(ImageClassifierBase):

    def __init__(self, input_shape, lr, mc=True, metrics=True, trunc=False, trained_model=None, model_name="test"):
        super().__init__(input_shape, lr, mc, metrics, trunc, trained_model, model_name)

    def _feature_extraction(self, inputs):
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPool2D(pool_size=(2, 2))(conv1)
        conv1 = self._get_dropout(conv1, rate=0.2)

        conv2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        conv2 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPool2D(pool_size=(2, 2))(conv2)
        conv2 = self._get_dropout(conv2, rate=0.2)

        conv3 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        conv3 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = MaxPool2D(pool_size=(2, 2))(conv3)
        conv3 = self._get_dropout(conv3, rate=0.2)

        output_list = [conv1, conv2, conv3]

        return output_list

    def _classifier(self, concatenated_features):
        x = Flatten()(concatenated_features)
        x = Dense(units=128, activation='relu')(x)

        if not self.trunc:
            x = self._get_dropout(x, rate=0.7)
            x = Dense(units=64, activation='relu')(x)
            x = self._get_dropout(x, rate=0.5)
            x = Dense(3, activation='softmax')(x)

        return x
