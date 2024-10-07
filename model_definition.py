import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json


class SegmentationModel:
    def __init__(self, input_shape=(256, 256, 3)):
        self.model = self.build_unet(input_shape)

    @staticmethod
    def conv_block(inputs, num_filters):
        x = Conv2D(num_filters, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    @staticmethod
    def encoder_block(input, num_filters):
        x = SegmentationModel.conv_block(input, num_filters)
        p = MaxPool2D((2, 2))(x)
        return x, p

    @staticmethod
    def decoder_block(input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = SegmentationModel.conv_block(x, num_filters)
        return x

    def build_unet(self, input_shape):
        inputs = Input(input_shape)
        # Encoder
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        # Bridge
        b1 = self.conv_block(p4, 1024)

        # Decoder
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        model = Model(inputs, outputs, name="UNet")
        return model
    
    
    def preprocess_image(image, target_size):
        image = cv2.resize(image, target_size)  # Resize to target size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = image / 255.0  # Normalize the image to [0, 1]
        return np.expand_dims(image, axis=0)  # Add batch dimension
    # Interpoltion Function

    
    @staticmethod
    def fill_image(img_masked, i_inds, j_inds, interp_win=10):
        for i, j in zip(i_inds, j_inds):
            if (i>=interp_win) or (j>=interp_win):
                template = img_masked[i: i+2*interp_win, j: j+2*interp_win].flatten()
            elif (img_masked.shape[0] - i < interp_win) or (img_masked.shape[1] - j < interp_win):
                template = img_masked[i-2*interp_win: i, j-2*interp_win: j].flatten()
            else:
                template = img_masked[i-interp_win: i+interp_win, j-interp_win: j+interp_win].flatten()

            template = np.delete(template, (template < 0))
            img_masked[i, j] = np.median(template)
        return img_masked
    
    @staticmethod
    def fill_line(x, y, step=1):
        points = []
        if x[0] == x[1]:
            ys = np.arange(y.min(), y.max(), step)
            xs = np.repeat(x[0], ys.size)
        else:
            m = (y[1] - y[0]) / (x[1] - x[0])
            xs = np.arange(x[0], x[1], step * np.sign(x[1]-x[0]))
            ys = y[0] + m * (xs-x[0])
        return xs.astype(int), ys.astype(int)
    
    @staticmethod
    def remove_hair_with_mask(image, mask):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize mask to match image size
        mask = mask / 255
        mask = (mask > 0.07).astype(np.uint8)

        # Use inpainting to fill in the masked regions
        inpaint_radius = 7
        inpaint_method = cv2.INPAINT_NS  # cv.INPAINT_NS or INPAINT_TELEA
        hair_free_image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)
        hair_free_image = cv2.inpaint(hair_free_image, mask, inpaint_radius, inpaint_method)

        return hair_free_image
    def compile(self, optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        return self.model.summary()

