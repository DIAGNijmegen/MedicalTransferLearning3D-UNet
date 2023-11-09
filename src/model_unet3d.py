import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Conv3D, Concatenate, Cropping3D, UpSampling3D, \
    Activation, Conv3DTranspose, Softmax, MaxPooling3D

tf.keras.backend.set_image_data_format('channels_first')


def unet_cropping(first_block=None, net=None):
    """ Function that wraps the Cropping3D layer to crop layers from different
        shape.
    """
    # net_shape = (int(net._keras_shape[2]), int(net._keras_shape[3]), int(net._keras_shape[4]))
    net_shape = (int(net.shape[2]), int(net.shape[3]), int(net.shape[4]))
    first_block_shape = (int(first_block.shape[2]), int(first_block.shape[3]), int(first_block.shape[4]))
    cropping = tuple(np.asarray(first_block_shape) - np.asarray(net_shape))
    first_block = Cropping3D(cropping=((int(cropping[0] / 2), int(cropping[0] / 2)),
                                       (int(cropping[1] / 2), int(cropping[1] / 2)),
                                       (int(cropping[2] / 2), int(cropping[2] / 2))))(first_block)

    return first_block


class UNet3DModel:
    """ Class that defines the 3D U-Net model. The function convnet_settings shows
        the parameters that are necessary to build the 3D U-Net.
    """
    def __down_path(self, tensor_in, nfilters, nonlinearity, initializer, name=None):
        """ Builds a block of the encoder path of 3D U-Net.
        """
        kernel_regularizer = None if not self.do_l2 else regularizers.l2(0.00003)
        convd = Conv3D(nfilters, (3, 3, 3), activation=nonlinearity,
                padding='valid', kernel_initializer=initializer,
                kernel_regularizer=kernel_regularizer)(tensor_in)
        convd = Conv3D(nfilters * 2, (3, 3, 3), activation=nonlinearity,
                padding='valid', kernel_initializer=initializer,
                name=name)(convd)

        return convd

    def __up_path(self, tensor_in, nfilters, nonlinearity, initializer):
        """ Builds a block of the decoder path of 3D U-Net.
        """
        kernel_regularizer = None if not self.do_l2 else regularizers.l2(0.00003)
        convu = Conv3D(nfilters, (3, 3, 3), activation=nonlinearity, 
                padding='valid', kernel_initializer=initializer, 
                kernel_regularizer=kernel_regularizer)(tensor_in)
        convu = Conv3D(nfilters, (3, 3, 3), activation=nonlinearity,
                padding='valid', kernel_initializer=initializer)(convu)

        return convu

    @staticmethod
    def __concat_up(tensor_up, tensor_crop):
        """ Upsamples, crops layers from different shapes, and concatenates them.
        """
        convup = UpSampling3D(size=2)(tensor_up)
        convc = unet_cropping(first_block=tensor_crop, net=convup)
        convc = Concatenate(axis=1)([convup, convc])

        return convc

    def __build_3d_unet(self, nonlinearity, num_classes=2, nfilters=32, initializer='glorot_uniform'):
        """ Returns the model of a 3D U-Net Cicek
            https://arxiv.org/abs/1606.06650
        """
        conv1 = self.__down_path(self.inputs, nfilters, nonlinearity, initializer)
        conv = MaxPooling3D(pool_size=self.pool_size)(conv1)

        conv2 = self.__down_path(conv, nfilters * 2, nonlinearity, initializer)
        conv = MaxPooling3D(pool_size=self.pool_size)(conv2)

        conv3 = self.__down_path(conv, nfilters * 4, nonlinearity, initializer)
        conv = MaxPooling3D(pool_size=self.pool_size)(conv3)

        conv = self.__down_path(conv, nfilters * 8, nonlinearity, initializer)

        conv = self.__concat_up(conv, conv3)
        conv = self.__up_path(conv, nfilters * 8, nonlinearity, initializer)

        conv = self.__concat_up(conv, conv2)
        conv = self.__up_path(conv, nfilters * 4, nonlinearity, initializer)

        conv = self.__concat_up(conv, conv1)
        conv = self.__up_path(conv, nfilters * 2, nonlinearity, initializer)

        output = Conv3D(num_classes, (1, 1, 1), padding='valid', kernel_initializer=initializer, name='output_segm')(conv)
        output = Softmax(axis=1)(output)
        model = Model(inputs=self.inputs, outputs=output)

        return model

    def convnet_settings(self, cube_dim, numclasses, num_filters=32,
                         lr_segm=0.00001, nonlinearity='relu', do_l2=True,
                         fine_tuning=False, initializer='glorot_uniform'):
        """ Function that returns the compiled 3D U-Net.
            - cube_dim: The input to the network receives cubes, this parameter
                        is a Integer value that represents the size of the 
                        cube. Valid values are 108, 132, 156, 180, and so on.
                        The larger the cube_dim, the more memory the model will
                        need on the GPU.
            - numclasses: Number of classes that the model will predict. This
                        model requires the background to be included on the
                        numclasses. For instance, to segment the liver, the
                        numclasses will be 2, one class for the liver and
                        another for the non-liver class (background). To 
                        segment the liver and the spleen, the numclasses will
                        be 3, one class for the liver, one class for the
                        spleen,and one class for the background.
            - num_filters: 32 is the default number defined on the 3D U-Net
                        paper. This parameter defines the number of filters of
                        the first block of convolutions. The number of filters
                        increases on the encoder path of the network.
            - lr_segm   : Learning rate. By default, this function uses Adam as
                        optimizer.
            - nonlinearity: Default value is 'relu'.
            - do_l2:    : To use L2.
            - fine_tuning: This option, when enabled, freezes the layers and
                        allows training on the last three layers of the network
                        for fine tuning. Once the model was fine tuned, it is
                        suggested to train the optimal model without fine
                        tuning for a better performance.
            - initializer: Function that initializes the weights of the
                        convolutions.
        """
        assert cube_dim in [108, 132, 156, 180]
        self.do_l2 = do_l2
        input_shape = (1, cube_dim, cube_dim, cube_dim)
        self.inputs = Input(input_shape)
        self.lr_segm = lr_segm

        model = self.__build_3d_unet(nonlinearity, num_classes=numclasses,
                                     nfilters=num_filters, initializer=initializer)

        if fine_tuning:
            for layer in model.layers[:-3]:
                print('Freezing layer', flush=True)
                layer.trainable = False

        # TODO: here define your loss and metrics of your preference
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr_segm),
                loss='categorical_crossentropy', metrics=['acc'])
        model.summary(line_length=115)
        return model

    def __init__(self):
        self.inputs = None
        self.pool_size = (2, 2, 2)


def get_3d_unet(cube_dim, n_classes):
    """ It creates a 3D UNet model based on the parameters. See the documentation of 
        convnet_settings for more detailed description of the parameters.
    """
    unetobj = UNet3DModel()
    unet_model = unetobj.convnet_settings(cube_dim, n_classes, 32, False, 
            do_l2=True)

    return unet_model


if __name__ == "__main__":
    cube_dim = 132
    n_classes = 2
    unet_model = get_3d_unet(cube_dim, n_classes)

    # 88 represents the receptive field of the 3D U-Net network
    pred_dim = cube_dim - 88
    input_cube = np.zeros((1, 1, cube_dim, cube_dim, cube_dim))
    output = unet_model.predict(input_cube)
    assert output.shape == (1, n_classes, pred_dim, pred_dim, pred_dim)
