# Deep GrabCut (DeepGC)

![DEXTR](doc/deepgc.png)

This is a PyTorch implementation of [Deep GrabCut](https://arxiv.org/pdf/1707.00243), for object segmentation.

### Installation
The code was tested with Python 3.5. To use this code, please do:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/DeepGrabCut-PyTorch
    cd DeepGrabCut-PyTorch
    ```
 
1. Install dependencies:
    ```Shell
    pip install pytorch torchvision -c pytorch
    pip install matplotlib opencv pillow
    ```
  
2. You can download pretrained model from [here](), and then put the model into models/

3. To try the demo of Deep GrabCut, please run:
    ```Shell
    python demo.py
    ```
If installed correctly, the result should look like this:
<p align="center"><img src="doc/demo.gif" align="center" width=450 height=auto/></p>

To train Deep GrabCut on PASCAL (or PASCAL + SBD), please follow these additional steps:

4. Download the pre-trained PSPNet model for semantic segmentation, taken from [this](https://github.com/isht7/pytorch-deeplab-resnet) repository.
    ```Shell
    cd models/
    chmod +x download_pretrained_psp_model.sh
    ./download_pretrained_psp_model.sh
    cd ..
    ```
5. Set the paths in ```mypath.py```, so that they point to the location of PASCAL/SBD dataset.

6. Run ```python train.py``` to train Deep Grabcut.

### Acknowledgment
We thank the authors of [DEXTR-PyTorch](https://github.com/scaelles/DEXTR-PyTorch), our repo borrows tons of code from they.

