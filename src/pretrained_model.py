import numpy as np
import model_unet3d


def load_sparse_pretrained_3d_unet(path_model, cube_dim):
    """ It loads the model and weights of the pretrained model used
        on the paper:
        'Transfer learning blah blah'
        by G. Humpire, C. Jacobs?, M. Prokop?, B. van Ginneken?, and N. Lessman?.
        This pretrained model contains 23 classes, 22 are structures of the
        body and one class for the background.
    """
    classes = 23
    unet_model = model_unet3d.get_3d_unet(cube_dim, classes)
    unet_model.load_weights(path_model)

    return unet_model


if __name__ == "__main__":
    path_model = '../weights/unet3d_Isomulti1v810.h5'
    cube_dim = 132

    unet_model = load_sparse_pretrained_3d_unet(path_model, cube_dim)

    # 88 represents the receptive field of the 3D U-Net network
    pred_dim = cube_dim - 88
    input_cube = np.zeros((1, 1, cube_dim, cube_dim, cube_dim))
    output = unet_model.predict(input_cube)
    assert output.shape == (1, 23, pred_dim, pred_dim, pred_dim)
