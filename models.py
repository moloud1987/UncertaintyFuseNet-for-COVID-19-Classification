import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

image_size = 150


def get_dropout(input_tensor, rate, mc=False):
    if mc:
        return Dropout(rate=rate)(input_tensor, training=True)
    else:
        return Dropout(rate=rate)(input_tensor)


def fusion_model(mc, lr=0.00005):
    inputs = Input(shape=(image_size, image_size, 1))
    input2 = tf.stack([inputs, inputs, inputs], axis=3)[:, :, :, :, 0]
    vgg_model = tf.keras.applications.VGG16(weights='imagenet',
                                            include_top=False,
                                            input_shape=(image_size, image_size, 3))
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
    conv4 = get_dropout(conv4, rate=0.2, mc=mc)

    # Fifth conv block
    conv5 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    conv5 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = MaxPool2D(pool_size=(2, 2))(conv5)
    conv5 = get_dropout(conv5, rate=0.2, mc=mc)

    concatenated_tensor = Concatenate(axis=1)(
        [Flatten()(conv3), Flatten()(conv4), Flatten()(conv5), Flatten()(vgg_feature)])

    # FC layer
    x = Flatten()(concatenated_tensor)
    x = Dense(units=512, activation='relu')(x)

    x = get_dropout(x, rate=0.7, mc=mc)
    x = Dense(units=128, activation='relu')(x)
    x = get_dropout(x, rate=0.5, mc=mc)
    x = Dense(units=64, activation='relu')(x)
    x = get_dropout(x, rate=0.3, mc=mc)
    # Output layer
    output = Dense(3, activation='softmax')(x)

    METRICS = [
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')]

    # Creating model and compiling
    model = Model(inputs=inputs, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', METRICS])

    # Callbacks
    if mc:
        mcheck = ModelCheckpoint('model_covid_mc.h5', monitor='val_accuracy', mode='max', verbose=1,
                                 save_best_only=True)
    else:
        mcheck = ModelCheckpoint('model_covid_simple.h5', monitor='val_accuracy', mode='max', verbose=1,
                                 save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, verbose=1, patience=5)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=30)
    callbacks = [reduce_lr, es, mcheck]

    return model, callbacks
