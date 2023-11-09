# Transfer learning using 3D U-Net for medical imaging

This repository contains the model used for the paper ''Transfer learning from a sparsely annotated dataset of 3D medical images'' by G. Humpire, C. Jacobs, M. Prokop, B. van Ginneken, N. Lessmann.

## Input

### Input size
The usual input size for a 3D U-Net is `132 x 132 x 132`, this can be defined by setting the parameter `cube_dim` to 132. The full list of allowed `cube_dim` is: 108, 132, 156, and so on. Since the receptive field of the 3D U-Net is 88, the output for an input `132 x 132 x 132` will be `44 x 44 x 44`.

### Image orientation
This network was trained using the image orientation Z, Y, X. We recommend using SimpleITK’s ImageFileReader to read the medical images. Other image orientations may not work as expected.

### Input value range
The input to the network must be within the [-500, 400] Hounsfield Unit range. Values outside that range must be clipped.

## Structure of this repository
- `src/`
  - `notebook.ipynb` Shows how to use the code in this repository.
  - `model_unet3d.py.py` defines the network (`class UNet3DModel`) in Keras.
    - Function `get_3d_unet` loads a randomly initialized 3D U-Net. It receives two parameters, `cube_dim` and `n_classes`. Note that `n_classes` must count in the background class.
  - `pretrained_model.py`:
    - Function `load_sparse_pretrained_3d_unet` gets the network and the weights of the network, which was trained on the sparsely annotated dataset.
  - `kerasmodel_to_onnx.py.py` loads the weights of the 3D U-Net model and transforms the Keras model to ONNX format.
  - `change_num_classes.py` loads the weights of the sparsely trained 3D U-Net (23 classes) into a 3D U-Net with a different number of classes. This script does not initialize the head with random weights, instead, it re-utilizes the weights of the pretrained model.
- `weights/` Contains the weights of the model in keras and ONNX format.
- `DATASOURCES.md` Contains a list of datasets used in our paper.
- `requirements.txt` List of packages and their versions.

## How to use
See `src/notebook.ipynb` for examples of how to execute the code of this repository.

## Index of weights
Follows the list of structures and their corresponding index in the output of the model.

| Index class | Structure |
| -- | -- |
| 0 | Aorta |
| 1 | Bladder |
| 2 | Duodenum |
| 3 | Gallbladder |
| 4 | Heart |
| 5 | Inferior vena cava |
| 6 | Left adrenal gland |
| 7 | Left kidney |
| 8 | Left lung |
| 9 | Left psoas major |
| 10 | Liver |
| 11 | Pancreas |
| 12 | Portal vein and splenic vein |
| 13 | Right adrenal gland |
| 14 | Right kidney |
| 15 | Right lung |
| 16 | Right psoas muscle |
| 17 | Spinal cord |
| 18 | Spleen |
| 19 | Sternum |
| 20 | Stomach |
| 21 | Trachea |
| 22 | **Background** |

The network was trained using the channels first format, then, the output of the network with batch size one is:

`1 x 23 x cube_dim x cube_dim x cube_dim`

## Compatibility
Please check the file `requirements.txt` to verify the minimum versions of the packages needed to run our scripts.
