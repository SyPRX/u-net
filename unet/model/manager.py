import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, \
    Conv2D, Add, concatenate, Cropping2D, UpSampling2D, Activation, BatchNormalization, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
import os
import shutil


class Model_manager:
    """
    The Model manager provided methods and attributes to manage a tf.keras.models.Model
    """

    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.model = None
        self.callbacks = None
        self.block_type = {
            "standard": self.block_standard,
            "residual": self.block_residual,
            "rrdb": self.block_residual_in_residual
        }
        pass

    def block_standard(self, input,feature_map = 64, conv_nb=2, bn=False, padding="same", kernel_size=3):
        """
        Define the block as described in the original U-net paper
        :param input: input layer
        :param feature_map: number of feature map
        :param conv_nb: number of convolution
        :param bn: Boolean for batch normalisation
        :param padding: padding type
        :param kernel_size: conv filter size
        :return: block output layer
        """
        layer = input
        for i_conv in range(conv_nb):
            layer = Conv2D(feature_map, kernel_size, padding=padding, kernel_initializer='he_normal')(layer)
            if bn:
                layer = BatchNormalization()(layer)
            layer = Activation("relu")(layer)
        return layer

    def block_residual(self, input, feature_map = 64, conv_nb=2, bn=False, padding="valid"):
        """
        Define a Residual block
        :param input: input layer
        :param feature_map: number of feature map
        :param conv_nb: number of convolution
        :param bn: Boolean for batch normalisation
        :param padding: padding type
        :return: block output layer
        """
        # Overwriting value to keep dimension
        padding = "same"
        input = Conv2D(feature_map, 1, padding=padding, kernel_initializer='he_normal')(input)
        layer = self.block_standard(input,feature_map = feature_map, conv_nb=conv_nb, bn=bn, padding=padding)
        return Add()([input, layer])

    def block_residual_in_residual(self, input, feature_map = 64, conv_nb=2, bn=False, padding="valid"):
        """
        Define the Residual in Residual block proposed in the ESRGAN paper
        :param input: input layer
        :param feature_map: number of feature map
        :param conv_nb: number of convolution
        :param bn: Boolean for batch normalisation
        :param padding: padding type
        :return: block output layer
        """
        # 4 residual block
        padding = "same"
        input = Conv2D(feature_map, 1, padding=padding, kernel_initializer='he_normal')(input)
        layer = input
        for i in range(4):
            layer = self.block_residual(layer, feature_map=feature_map, conv_nb=conv_nb, bn=bn, padding=padding)
        return Add()([input, layer])

    def get_callbacks(self):
        """
        Define model callbacks (Model check point and tensorflow logs)
        :return: model callbacks
        """
        log_file = 'data/log.tf'
        if os.path.isdir(log_file):
            shutil.rmtree(log_file)
            print("Removing logs before training")
        tensorboard = TensorBoard(log_dir=log_file)

        # Saving multiple model during training
        if not os.path.isdir("data/models"):
            os.mkdir("data/models")
        model_file = "data/models/unet_model.{epoch:02d}-{val_loss:.4f}.hdf5"
        model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=False)

        weight_file = "data/models/unet_weight.{epoch:02d}-{val_loss:.4f}.hdf5"
        weight_checkpoint = ModelCheckpoint(weight_file, monitor='loss', verbose=1, save_weights_only=True)

        return [tensorboard, model_checkpoint, weight_checkpoint]

    def create_model(self,
                     img_size=(256, 256),
                     pooling_nb=4,
                     conv_nb=2,
                     block_type="standard",
                     pretrained_weights=None,
                     padding="same",
                     batch_norm=False):
        """
        Define u-net model architecture regarding few
        :param img_size: image input size
        :param pooling_nb: number of pooling operation
        :param conv_nb: number of convolutional layer within a block
        :param block_type: block type, standard, residual etc...
        :param pretrained_weights: pretrained weight provided for initialisation
        :param padding: padding type
        :param batch_norm: Boolean for batch normalisation
        :return: tf.keras.models.Model
        """

        # Defining Input
        inputs = Input((img_size[0], img_size[1], 1))


        layer = inputs
        feature_map = 64
        downsampler_layers = []

        # Encoder steps (Down sampling)

        for i in range(pooling_nb):
            layer = self.block_type[block_type](layer,conv_nb=conv_nb, feature_map=feature_map,bn=batch_norm, padding=padding)
            if i == (pooling_nb-1):
                layer = Dropout(0.5)(layer)
            downsampler_layers.append(layer)
            layer = MaxPooling2D(pool_size=(2, 2))(layer)
            feature_map *= 2

        # Bridge
        layer = self.block_type[block_type](layer, conv_nb=conv_nb, feature_map=feature_map, bn=batch_norm, padding=padding)
        layer = Dropout(0.5)(layer)

        # Decoder steps (Up sampling)

        for i in range(pooling_nb)[::-1]:
            # Up sampling and conv
            feature_map = int(feature_map / 2)
            layer = UpSampling2D(size=(2, 2))(layer)
            layer = Conv2D(feature_map, 2, activation='relu', padding="same", kernel_initializer='he_normal')(layer)

            # Cropping along x and y the encoder layer
            crop_x = None
            if ((downsampler_layers[i].shape[1] - layer.shape[1]) % 2) == 0:
                crop_x = [int((downsampler_layers[i].shape[1] - layer.shape[1])/2)]*2
            else:
                crop_x = (int((downsampler_layers[i].shape[1] - layer.shape[1])/2), int((downsampler_layers[i].shape[1] - layer.shape[1])/2)+ 1)
            crop_y = None
            if ((downsampler_layers[i].shape[2] - layer.shape[2]) % 2) == 0:
                crop_y = [int((downsampler_layers[i].shape[2] - layer.shape[2])/2)]*2
            else:
                crop_y = (int((downsampler_layers[i].shape[2] - layer.shape[2])/2), int((downsampler_layers[i].shape[2] - layer.shape[2])/2)+ 1)
            cropped_layer = Cropping2D(cropping=(crop_x, crop_y))(downsampler_layers[i])

            # Stacking both encoder an decoder layers
            layer = concatenate([cropped_layer, layer], axis=3)

            layer = self.block_type[block_type](layer, feature_map=feature_map, bn=batch_norm, padding=padding)

        # Final stage
        layer = Conv2D(2, 3, activation='relu', padding=padding, kernel_initializer='he_normal')(layer)
        outputs = Conv2D(1, 1, activation='sigmoid')(layer)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        self.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        if pretrained_weights:
            print("Loading weights from {}".format(pretrained_weights))
            self.model.load_weights(pretrained_weights)

        return self.model

    def get_model(self, mode_filepath = None):
        """
        Loading a specific model
        :param mode_filepath: model file path if provided
        :return: tf.keras.models.Model
        """
        if mode_filepath is not None:
            assert  os.path.isfile(mode_filepath), "[MODEL ERROR]: The model filepath {} does not exist. " \
                                                   "Please trained a model before predicting or specify " \
                                                   "a valid filepath".format(mode_filepath)

            self.model = load_model(mode_filepath)
        else:
            print("Choose one of those models in order to make some predictions")
            models_filepathes = self.list_model_filenames(folder=self.model_folder)
            for index, file in enumerate(models_filepathes):
                print(index, ":", file)
            resp = int(input("Enter the model index you want to load :\n"))
            mode_filepath = models_filepathes[resp]
            print("Loading model : {}".format(mode_filepath))
            self.model = load_model(os.path.join(self.model_folder,mode_filepath))

        return self.model

    def list_model_filenames(self, folder ="data/models"):
        """
        List model file in a specific folder
        :param folder: model folder
        :return: list of model filenames
        """
        models_filenames = []
        for file in os.listdir(folder):
            if "model" in file:
                models_filenames.append(file)
        return models_filenames
