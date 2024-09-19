import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, GlobalAveragePooling2D, \
    Dense, Concatenate, Input,Flatten,Dropout,MaxPooling2D
from keras.optimizers import SGD



class DenseNet121:

    def __init__(self,X, input_shape=(224, 224, 3), num_classes=5):
        self.learning_rate =X[0]
        self.Dropout_rate=X[1]
        self.compression_rate=0.5
        self.model = self.run(input_shape, num_classes)
    def dense_block(self, inputs, num_layers, num_input_features,
                                bn_size, growth_rate):
        for i in range(num_layers):
            x = BatchNormalization()(inputs)
            x = Activation('relu')(x)
            x = Conv2D(num_input_features + i * growth_rate, (1, 1), padding='same', strides=1, kernel_initializer='he_normal',
                       use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(growth_rate * bn_size, (3, 3), padding='same', strides=1, kernel_initializer='he_normal', use_bias=False)(x)
            inputs = Concatenate()([inputs, x])

        return inputs

    def transition_layer(self, x, num_output_features):
        """Transition layer between two adjacent DenseBlock"""
        # BatchNormalization
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # Convolution
        x = Conv2D(num_output_features, (1, 1),strides=1, padding='same',use_bias=False)(x)
        shape = x.get_shape().as_list()
        if shape[1] > 2 and shape[2] > 2:
            x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x

    def run(self, input_shape, num_classes,num_init_features=64,bn_size=4,block_config=(6, 12, 24, 16),growth_rate=32):
        inputs = Input(shape=input_shape)
        x = Conv2D(num_init_features*2, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal',use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3,3), strides=(2, 2),padding='same')(x)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            x = self.dense_block(x,num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate)
            num_features = num_features + num_layers * growth_rate
            if i!=len(block_config)-1:
                x = self.transition_layer(x, self.compression_rate*num_features)
                num_features//=2

        # final bn+ReLU
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.Dropout_rate)(x)
        # classification layer
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        return model

    def model_create(self, learning_rate):
        self.model.compile(optimizer=SGD(learning_rate=learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        return self.model