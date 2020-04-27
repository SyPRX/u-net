from unet import data
from unet import model
import unet.utils.functions as gf
import argparse
import os
import math
import numpy as np
from PIL import Image
class Unet:
    """
    The Unet class implement the model architecture as depicted in the original paper
    """

    def __init__(self, ):

        # # Parsing argument
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode",
                            default="train",
                            choices=["train", "predict_test", "predict"],
                            type=str,
                            help="Either 'train' or 'predict'")
        parser.add_argument("--weights", default=None, type=str, help="filepath to pretrained weights")
        parser.add_argument("--pooling", default=4, type=int, help="Number of pooling operation")
        parser.add_argument("--conv", default=2, type=int, help="Number of convolutional layer")
        parser.add_argument("--img_size", default=256, type=int, help="Input image size")
        parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
        parser.add_argument("--step_per_epoch", default=300, type=int, help="Step per epoch")
        parser.add_argument("--epoch", default=1, type=int, help="Number of epoch")
        parser.add_argument("--input", default="data/predict_input.png", type=str, help="Default image or image folder "
                                                                                        "to predict")
        parser.add_argument("--model_file", default=None, type=str, help="Model to load for prediction")
        parser.add_argument("--padding", default="same", choices=["same", "valid"], type=str, help="Default padding")
        parser.add_argument("--block_type",
                            default="standard",
                            choices=["standard", "residual", "rrdb"],
                            type=str,
                            help="Block type")
        parser.add_argument("--dataset",
                            default="EM",
                            choices=["EM"],
                            type=str,
                            help="Dataset name supported")
        parser.add_argument("--batch_norm", type=gf.str2bool, nargs='?',
                            const=True, default=False,
                            help="Add batch normalisation layers.")

        args = parser.parse_args()

        # PARAMS
        self.MODE = args.mode
        self.POOLING = args.pooling
        self.CONV = args.conv
        self.PRETRAINED = args.weights
        self.INPUT = args.input
        self.PADDING = args.padding
        self.BLOCK_TYPE = args.block_type
        self.IMG_SIZE = (args.img_size, args.img_size)
        self.BATCH_SIZE = args.batch_size
        self.DATASET_NAME = args.dataset
        self.STEP_PER_EPOCH = args.step_per_epoch
        self.EPOCH = args.epoch
        self.MODEL_FILE = args.model_file
        self.BATCH_NORM = args.batch_norm

        # Assertion
        assert self.POOLING >= 0, "Pooling number should be positive"
        assert self.CONV >= 0, "The number of convolution should be positive"
        assert self.IMG_SIZE[0] > 0, "Image size should be high enough"
        assert self.BATCH_SIZE > 0, "Batch size should be positive"
        assert self.STEP_PER_EPOCH > 0, "Step per epoch should be positive"
        assert self.EPOCH > 0, "Number of epoch should be positive"
        assert self.POOLING <= math.log(self.IMG_SIZE[0]) / math.log(
            2), "Pooling number exceeds regarding input img size"  # assuming padding is same

        # Checking EM dataset
        assert os.path.isfile("data/train-volume.tif"), "The EM dataset files do not exist"
        assert os.path.isfile("data/train-labels.tif"), "The EM dataset files do not exist"
        assert os.path.isfile("data/test-volume.tif"), "The EM dataset files do not exist"

        # DATA
        self.data_manager = data.Data_manager(batch_size=self.BATCH_SIZE, img_size=self.IMG_SIZE)

        # MODEL
        self.model_folder = "data/models"
        self.model_manager = model.Model_manager(model_folder=self.model_folder)
        self.model = None
        self.callbacks = None

        # Checking models for predictions
        if "predict" in self.MODE:
            if self.MODEL_FILE is not None:
                assert os.path.isfile(self.MODEL_FILE), "please specify a valid model filepath"
            else:
                assert os.path.isdir(
                    self.model_folder), "No model file available, please train a model before inference"
                assert len([file for file in os.listdir(self.model_folder) if
                            "model" in file]) > 0, "No model file available, please train a model before inference"

        # Create tmp folder
        self.tmp_folder = "data/tmp"
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)

        # Functions:
        self.pipeline = {"train": self.train,
                         "predict": self.predict,
                         "predict_test": self.predict_test}

    def run(self):
        # Calling the corresponding pipeline function
        self.pipeline[self.MODE]()

    def train(self):
        """
        Train function of the U-net project. It implements training pipeline.
        :return:
        """

        # Load model
        self.model = self.model_manager.create_model(img_size=self.IMG_SIZE,
                                                     pooling_nb=self.POOLING,
                                                     padding=self.PADDING,
                                                     block_type=self.BLOCK_TYPE,
                                                     pretrained_weights=self.PRETRAINED,
                                                     batch_norm=self.BATCH_NORM)

        # Set output size regarding model architecture
        self.data_manager.output_size = (
            self.model.layers[-1].output_shape[1],
            self.model.layers[-1].output_shape[2]
        )

        # Loading datasets
        dataset_train, dataset_validation, dataset_test = self.data_manager.get_dataset(
            dataset_name=self.DATASET_NAME)

        self.callbacks = self.model_manager.get_callbacks()

        # Fit train generator
        self.model.fit(dataset_train,
                       validation_data=dataset_validation,
                       validation_steps=self.STEP_PER_EPOCH,
                       steps_per_epoch=self.STEP_PER_EPOCH,
                       epochs=self.EPOCH,
                       verbose=1,
                       callbacks=self.callbacks)

    def predict(self):
        """
        Inference pipeline for one or few images
        """
        # Load model
        self.model = self.model_manager.get_model(mode_filepath=self.MODEL_FILE)

        # Get model's input shape
        self.IMG_SIZE = (self.model.layers[0].input_shape[0][1], self.model.layers[0].input_shape[0][2])
        self.data_manager.img_size = self.IMG_SIZE

        # Loading image files
        input_data = None
        if os.path.isfile(self.INPUT):
            input_data = self.data_manager.load_image(self.INPUT)
        elif os.path.isdir(self.INPUT):
            input_data = self.data_manager.load_img_from_folder(self.INPUT)
        assert input_data is not None, "No valid img in this folder, subfolders are not supported"

        # Predict data
        predictions = self.model.predict(input_data, verbose=1)

        # Save predictions along input data for visual comparison
        self.data_manager.save_result(os.path.join(self.tmp_folder, "predictions"), predictions, input_data,
                                      batch_size=None)

    def predict_test(self):
        """
        Inference over the test set of the EM dataset
        :return:
        """
        # Load model
        self.model = self.model_manager.get_model(mode_filepath=self.MODEL_FILE)

        # Get model's input shape
        self.IMG_SIZE = (self.model.layers[0].input_shape[0][1], self.model.layers[0].input_shape[0][2])
        self.data_manager.output_size = (
            self.model.layers[-1].output_shape[1], self.model.layers[-1].output_shape[2])

        # Get datasets (especially test dataset)
        self.data_manager.img_size = self.IMG_SIZE
        dataset_train, dataset_validation, dataset_test = self.data_manager.get_dataset(
            dataset_name=self.DATASET_NAME)

        # Predict generator
        predictions = self.model.predict(dataset_test, verbose=1)

        # Save predictions along input data for visual comparison
        self.data_manager.save_result(os.path.join(self.tmp_folder, "test"), predictions, dataset_test,
                                      batch_size=self.BATCH_SIZE)
