import os
import cv2
import numpy as np
from pretrained_model import load_sparse_pretrained_3d_unet


class Unet:
    def __init__(self, weights):
        self.model = load_sparse_pretrained_3d_unet(weights, cube_dim=132)

    def __call__(self, patch):
        probabilities = self.model.predict(patch[None, None, :, :, :])
        foreground_predictions = probabilities[0, :-1, :, :, :] > 0.5
        predictions = np.pad(foreground_predictions, [(1, 0), (0, 0), (0, 0), (0, 0)])
        return np.argmax(predictions, axis=0).astype('int16')


if __name__ == "__main__":
    #path_patch = 'cubes/liver-orig010__44_176_88.npy'
    path_patch = 'cubes/liver-orig010__132_176_44.npy'
    path_orthogonal_views_path = path_patch.replace(".npy", ".jpg")
    path_weights_h5 = '../weights/unet3d_Isomulti1v810.h5'

    # Load network
    network = Unet(path_weights_h5)

    # Load patch
    patch_in = np.load(path_patch)

    # Clips patch
    patch_in = np.clip(patch_in, -500, 400)

    # Gets prediction
    pred = network(patch_in).astype('uint8')

    # Visualization of predictions in axial, coronal, and sagittal views
    pred *= 10  # Makes the predictions brighter for visualization only
    hcat_patch = cv2.imread(path_orthogonal_views_path)
    hcat_patch = cv2.cvtColor(hcat_patch, cv2.COLOR_BGR2GRAY)
    hcat_overlap = np.copy(hcat_patch)
    hcat_overlap[44:88, 44*1:44*1+44] = pred[22, :, :]
    hcat_overlap[44:88, 44*4:44*4+44] = pred[:, 22, :]
    hcat_overlap[44:88, 44*7:44*7+44] = pred[:, :, 22]

    cv2.imshow("hcat_patch", hcat_patch)
    cv2.imshow("hcat_overlap", hcat_overlap)
    cv2.waitKey(0)
