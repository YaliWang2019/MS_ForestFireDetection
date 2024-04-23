# MS_ForestFireDetection
This is an archive of my graduate project. This project analyzes the feasibility of deploying machine learning models on a drone to detect forest fires.

Explanation for each folder:

## D2Go:
The D2Go model is developed based on Faster_RCNN.

1.  **D2GO_RCNN_PoygonSize.ipynb:** Script for detecting fires in an image and showing the bounding box sizes.
2.  **D2GO_RCNN_Video.ipynb:** Script for detecting fires in the frames of a video.
3.  **forest-fires-t.json and forest-fires-v.json:** Annotated COCO format JSON datasets for training the D2Go model.

## PyTorch_CNN:

The convolutional neural network models developed using PyTorch framework.

1.  **datasheets:** All the datasheets recorded during the tests.
2.  **models:**
 
- **pytorch_cnn_add1.pth:** 4 convolutional2D layers + 4 maxpooling2D layers + 3 dense layers;
- **pytorch_cnn_add2.pth:** 5 convolutional2D layers + 5 maxpooling2D layers + 3 dense layers;
-  **pytorch_cnn_origin.pth:** 3 convolutional2D layers + 3 maxpooling2D layers + 3 dense layers;
-  **pytorch_cnn_rm1.pth:** 2 convolutional2D layers + 2 maxpooling2D layers + 3 dense layers;
-  **pytorch_cnn_rm2.pth:** 1 convolutional2D layers + 1 maxpooling2D layers + 3 dense layers;
-  **pytorch_cnn_sig1.pth:** 3 convolutional2D layers + 3 maxpooling2D layers + 1 dense layers;
-  **pytorch_cnn_add2.pth:** 3 convolutional2D layers + 3 maxpooling2D layers + 2 dense layers;

3.  **scripts:** The training scripts of the models mentioned above and their energy consumption measurement scripts.

## TensorFlow_AE:

The Autoencoder models developed using TensorFlow framework.

1.  **datasheets:** All the datasheets recorded during the tests.
2.  **models:**

 -  **ae_1En_1De.h5:** 1 group of convolutional2D layer and maxpooling2D layer in Encoder + 1 group of convolutional2D layer and upsampling2D layer in Decoder;
 -  **ae_2En_2De.h5:** 2 groups of convolutional2D layer and maxpooling2D layer in Encoder + 2 groups of convolutional2D layer and upsampling2D layer in Decoder;
 -  **ae_3En_3De.h5:** 3 groups of convolutional2D layer and maxpooling2D layer in Encoder + 3 groups of convolutional2D layer and upsampling2D layer in Decoder;

3.  **scripts:** The training script of the original model (ae_3En_3De.h5) and its energy consumption measurement script.

## TensorFlow_CNN:

The convolutional neural network models developed using TensorFlow framework.

1.  **datasheets:** All the datasheets recorded during the tests.
2.  **models:**

 -  **cnn_or.pth:** 3 convolutional2D layers + 3 maxpooling2D layers + 3 dense layers;
 -  **cnn_rm1ConvMaxp.pth:** 2 convolutional2D layers + 2 maxpooling2D layers + 3 dense layers;
 -  **cnn_rm2ConvMaxp.pth:** 1 convolutional2D layers + 1 maxpooling2D layers + 3 dense layers;
 -  **cnn_sig1_dense.pth:** 3 convolutional2D layers + 3 maxpooling2D layers + 1 dense layers;
 -  **cnn_sig2_dense.pth:** 3 convolutional2D layers + 3 maxpooling2D layers + 2 dense layers;

3.  **scripts:** The training scripts of the original model (cnn_or.pth) and its energy consumption measurement script.

## TensorFlow_DBN:

The deep belief network models developed using TensorFlow models.

1.  **datasheets:** All the datasheets recorded during the tests.
2.  **models:** These four models can be accessed from [this google drive link](https://drive.google.com/drive/folders/1RRahxHpNXGpgZ7lKnLd5ulm3Rw1RE0bn?usp=sharing) since each of them is larger than 100 MB which makes GitHub unhappy to upload them here.

 -  dbn_1RBM.joblib: 1 Restricted Boltzmann Machine layer;
 -  dbn_2RBM.joblib: 2 Restricted Boltzmann Machine layers;
 -  dbn_3RBM.joblib: 3 Restricted Boltzmann Machine layers;
 -  dbn_4RBM.joblib: 4 Restricted Boltzmann Machine layers;

3.  **scripts:** The training scripts of the original model (dbn_2RBM.joblib) and its energy consumption measurement script.

 ## TensorFlow_UNet:

The U-Net models developed using TensorFlow framework.

1. **datasheets:** All the datasheets recorded during the tests.
2. **models:**

 -  **unet_1En_1De.h5:** 1 group of convolutional2D layer and maxpooling2D layer in Encoder + 1 group of convolutional2D layer, upsampling2D layer, and concatenate layer in Decoder;
 -  **unet_2En_2De.h5:** 2 groups of convolutional2D layer and maxpooling2D layer in Encoder + 2 groups of convolutional2D layer, upsampling2D layer, and concatenate layer in Decoder;
 -  **unet_3En_3De.h5:** 3 groups of convolutional2D layer and maxpooling2D layer in Encoder + 3 groups of convolutional2D layer, upsampling2D layer, and concatenate layer in Decoder;

3. **scripts:** The training scripts of the original model (unet_3En_3De.h5) and its energy consumption measurement script.
