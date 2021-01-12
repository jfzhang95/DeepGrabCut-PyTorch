# Deep GrabCut (DeepGC)

![DEXTR](doc/deepgc.png)

This is a PyTorch implementation of [Deep GrabCut](https://arxiv.org/pdf/1707.00243), for object segmentation. We use DeepLab-v2 instead of DeconvNet in this repository.

### Installation
The code was tested with Python 3.5. To use this code, please do:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/DeepGrabCut-PyTorch
    cd DeepGrabCut-PyTorch
    ```
 
1. Install dependencies:
    ```Shell
    pip install -r requirements.txt
    ```
  
2. Download pretained automatically. Or manually from [GoogleDrive](https://drive.google.com/open?id=1N8bICHnFit6lLGvGwVu6bnDttyTk6wGH), and then put the model into `models`. 
    ```Shell
    gdown --output ./models/deepgc_pascal_epoch-99.pth --id 1N8bICHnFit6lLGvGwVu6bnDttyTk6wGH
    ```

3. To try the demo of Deep GrabCut, please run:
    ```Shell
    python demo.py
    # 1-When window appears, press "s"
    # 2-Draw circle
    # 3-Press spacebar and wait for 2 - 3 seconds
    ```

If installed correctly, the result should look like this:
<p align="center"><img src="doc/demo.gif" align="center" width=450 height=auto/></p>
Note that the provided model was trained only on VOC 2012 dataset. You will get better results if you train model on both VOC and SBD dataset.

To train Deep GrabCut on VOC (or VOC + SBD), please follow these additional steps:

1. Download the pre-trained PSPNet model for semantic segmentation, taken from this [repository](https://github.com/isht7/pytorch-deeplab-resnet).
    ```Shell
    cd models/
    chmod +x download_pretrained_psp_model.sh
    ./download_pretrained_psp_model.sh
    cd ..
    ```
2. Set the paths in ```mypath.py```, so that they point to the location of VOC/SBD dataset.

3. Run ```python train.py``` to train Deep Grabcut.

4. If you want to train model on COCO dataset, you should first config COCO dataset path in mypath.py, and then run
```python train_coco.py``` to train model.
