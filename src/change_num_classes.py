import os
import numpy as np
import model_unet3d as unet3dtl
import pretrained_model


def get_model_n_classes_from_pretrained_model(src_model_path, dst_model_path, dst_net_classes):
    """ This funcion loads the weights of the pretrained model (23 classes) and
        saves a new network with N classes (dst_net_classes) for transfer
        learning. This function does not randomly initialize the new head 
        (dst_net_classes), instead, we reutilize the weights of our pretrained
        network and place them as the new head.
        - src_model_path: path of the pretrained model (.h5).
        - dst_model_path: path of the new model (.h5).
        - dst_net_classes: desired number of classes on the new network (including the background class).
    """
    src_net_classes = 23
    assert src_net_classes != dst_net_classes
    assert src_model_path != dst_model_path
    assert not os.path.exists(dst_model_path), dst_model_path + " already exists"
 
    # Creates an empty model with dst_net_classes as number of classes
    cube_dim = 132

    # Recreates the pretrained model. The number of classes should match dst_net_classes
    # Loads the weights
    model_pretrained_network = pretrained_model.load_sparse_pretrained_3d_unet(src_model_path, cube_dim)

    model_empty_network = unet3dtl.get_3d_unet(cube_dim, dst_net_classes)

    # Gets weights from the models
    weights_random = model_empty_network.get_weights()
    weights_pretrained = model_pretrained_network.get_weights()

    # This script will take the last dst_net_classes_empty classes,
    # including the background as the last class Weights kernel: 
    if src_net_classes < dst_net_classes:
        diff_classes = dst_net_classes - src_net_classes
        for idx in range(diff_classes):
            # This defines the index where the new class should be placed; then
            # at least in the new model the weights of the old background will
            # be the same background on the new model
            idx_new_class = src_net_classes + idx

            # Weights kernel: weights[-2] stands for the weights of the last layer
            weights_newclass_random = weights_random[-2][:, :, :, :, -(idx+1)]
            weights_pretrained[-2] = np.insert(weights_pretrained[-2], idx_new_class, weights_newclass_random, axis=4)

            # Bias: weights[-1] stands for the bias of the last layer
            bias_newclass_random = weights_random[-1][-(idx+1)]
            weights_pretrained[-1] = np.insert(weights_pretrained[-1], idx_new_class, bias_newclass_random, axis=0)

    elif src_net_classes > dst_net_classes:
        # Weights kernel: weights[-2] stands for the weights of the last layer
        weights_pretrained[-2] = weights_pretrained[-2][:, :, :, :, -dst_net_classes:]

        # Bias: weights[-1] stands for the bias of the last layer
        weights_pretrained[-1] = weights_pretrained[-1][-dst_net_classes:]

        # # This code takes weights from specific classes!! In this case Spleen (class 18)
        # tmp_weights = weights_pretrained[-2][:, :, :, :, 18:19]
        # tmp_weights = np.concatenate([tmp_weights, weights_pretrained[-2][:, :, :, :, -1:]], axis=4)
        # weights_pretrained[-2] = tmp_weights
        # tmp = [weights_pretrained[-1][18], weights_pretrained[-1][-1]]
        # weights_pretrained[-1] = np.array(tmp, dtype=np.float32)
    else:
        raise NotImplementedError('Both networks have the same number of classes')
    model_empty_network.set_weights(weights_pretrained)
    model_empty_network.save(dst_model_path)
    print('Output model:', dst_model_path)


if __name__ == '__main__':
    dst_net_classes = 5
    src_model_path = '../weights/unet3d_Isomulti1v810.h5'
    dst_model_path = '../weights/unet3d_{}_classes.h5'.format(dst_net_classes)
    assert os.path.exists(src_model_path)

    get_model_n_classes_from_pretrained_model(src_model_path, dst_model_path, 
                                              dst_net_classes)
