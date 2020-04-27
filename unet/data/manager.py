import tensorflow as tf
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import os
from unet.utils.functions import get_binary_mask

class Data_manager:
    """
    Data manager allows to manage data before, during and after training
    """
    def __init__(self, batch_size = 32, img_size = (256, 256) ):
        self.img_size = img_size
        self.output_size = None
        self.batch_size = batch_size
        self.supported_dataset = {"EM": self.load_em_dataset}
        pass

    def get_dataset(self, dataset_name = "EM"):
        """
        Get specific dataset using name
        :param dataset_name: Dataset name
        :return: tf.Dataset accordingly
        """
        return self.supported_dataset[dataset_name]()

    def preprocess_input(self, x: tf.Tensor) -> tf.Tensor:
        """
        Preprocessing input tensor and set size accordingly
        :param x: raw input tensor
        :return: preprocessed input tensor
        """
        # Scaling
        x /= 255.

        # Resizing input
        x = tf.image.resize(x, self.img_size)
        return x
    def preprocess_output(self, x: tf.Tensor) -> tf.Tensor:
        """
        Preprocessing ground truth and set size accordingly
        :param x: raw mask tensor
        :return: preprocessed mask tensor
        """
        # Scaling
        x /= 255.

        # Resize according to the model output size
        x = tf.image.resize(x, self.output_size)

        # Convert binary mask
        x = tf.numpy_function(get_binary_mask, [x], tf.float32)
        x = tf.reshape(x, self.output_size + (1,))

        return x
    def data_augmentation(self, img: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Few functions applied to the input image for data augmentation during training
        :param img: Input tensor
        :param mask: Groundtruth tensor
        :return: Augmented Inputs
        """

        img = self.preprocess_input(img)
        mask = self.preprocess_output(mask)

        # Random rotation
        param_1 = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, param_1)
        mask = tf.image.rot90(mask, param_1)

        # Flip left right
        do_flip = tf.random.uniform([]) > 0.5
        img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
        mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

        # Flip up down
        do_flip = tf.random.uniform([]) > 0.5
        img = tf.cond(do_flip, lambda: tf.image.random_flip_up_down(img), lambda: img)
        mask = tf.cond(do_flip, lambda: tf.image.random_flip_up_down(mask), lambda: mask)

        return img, mask

    def prepare_dataset(self, dataset, mode="train"):
        """
        Set specific parameters regarding the tf.Dataset class
        :param dataset: tf.Dataset
        :param mode: either train, validation or test
        :return: dataset prepared
        """
        if mode == "train" or mode == "validation":
            dataset = dataset.map(self.data_augmentation)
            dataset = dataset.repeat(None)
            dataset = dataset.shuffle(buffer_size=64)
        elif mode == "predict":
            dataset = dataset.map(self.preprocess_input)

        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


    def load_em_dataset(self,):
        """
        Loading the specific EM dataset
        :return: train, validation and test generator
        """
        # Load img and labels
        img = io.imread('data/train-volume.tif')
        img = np.expand_dims(img, axis=3)
        img = img.astype("float32")
        mask = io.imread('data/train-labels.tif')
        mask = np.expand_dims(mask, axis=3)
        mask = mask.astype("float32")

        # Split train and validation, 80 / 20
        train_img = img[:24]
        train_mask = mask[:24]

        valid_img = img[24:]
        valid_mask = mask[24:]

        dataset_train = tf.data.Dataset.from_tensor_slices((train_img, train_mask))
        dataset_validation = tf.data.Dataset.from_tensor_slices((valid_img, valid_mask))
        # Test
        test_img = io.imread('data/test-volume.tif')
        test_img = np.expand_dims(test_img, axis=3)
        # test_img = np.expand_dims(test_img, axis=0)
        test_img = test_img.astype("float32")
        dataset_test = tf.data.Dataset.from_tensor_slices(test_img)

        dataset_train = self.prepare_dataset(dataset_train, mode="train")
        dataset_validation = self.prepare_dataset(dataset_validation, mode="validation")
        dataset_test = self.prepare_dataset(dataset_test, mode="predict")

        # Caching the dataset for optimization
        dataset_train = dataset_train.cache("data/tmp/dataset_train.cache")
        dataset_validation = dataset_validation.cache("data/tmp/dataset_val.cache")
        # dataset_test = dataset_test.cache("data/tmp/dataset_test.cache")

        return dataset_train, dataset_validation, dataset_test

    def save_result(self, save_path, predictions, dataset_test, batch_size = None):
        """
        Saving predictions in images for visual check
        :param save_path: Folder to save the predicted images
        :param predictions: Inference results
        :param dataset_test: Input data
        :param batch_size: Boolean to check if the data has been processed by batch
        :return:
        """

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        count = 0

        for batch in dataset_test:
            if batch_size is not None:
                for img in batch:

                    input_img = img[:, :, 0]
                    pred_img = predictions[count][:, :, 0]

                    io.imsave(os.path.join(save_path, "%d_input.png" % count), input_img)
                    io.imsave(os.path.join(save_path, "%d_predicted.png" % count), pred_img)
                    count += 1
            else:

                input_img = batch[:, :, 0]
                pred_img = predictions[count][:, :, 0]

                io.imsave(os.path.join(save_path, "%d_input.png" % count), input_img)
                io.imsave(os.path.join(save_path, "%d_predicted.png" % count), pred_img)
                count += 1

    def is_valid_img(self, file_path):
        """
        Check either the file is an image file or not
        :param file_path: file path to the image file
        :return: Boolean
        """
        valid_extension = ("jpg", "jpeg", "png")
        if file_path.lower().endswith(valid_extension):
            return True
        return False

    def load_image(self, file_path):
        """
        Load and preprocess an image file
        :param file_path: file path to the image file
        :return: numpy array preprocessed
        """
        assert self.is_valid_img(file_path), "The image is not valid"
        test_img = io.imread(file_path)

        # Dealing with RGB
        if len(test_img.shape) == 3:
            if test_img.shape[2] == 3:
                test_img = rgb2gray(test_img) * 255.

        if len(test_img.shape) == 2:
            test_img = np.expand_dims(test_img, axis=2)

        test_img = np.expand_dims(test_img, axis=0)
        test_img = test_img.astype("float32")
        test_img = self.preprocess_input(test_img)
        return test_img

    def load_img_from_folder(self, folder):
        """
        Load image files and stack them in a numpy array
        :param folder: folder containing image files
        :return: numpy array preprocessed
        """
        stack = None
        for file in os.listdir(folder):
            if not self.is_valid_img(file):
                continue
            if stack is None:
                stack = self.load_image(os.path.join(folder, file))
            else:
                stack = np.vstack([stack, self.load_image(os.path.join(folder, file))])
        return stack