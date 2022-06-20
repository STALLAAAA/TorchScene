# TorchScene

A scene recognition tool based on pytorch

## Installation

```bash
conda env create -f environment.yml python=3.9

conda activate torch-scene

export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Model Zoo (Pretrained Models)

Please refer [[Model Zoo]](model_zoo.md)

## Step by step

Please download the data from [[Place2 Data]](http://places2.csail.mit.edu/download.html)


### 1. Download the data

These images are 256x256 images, in a more friendly directory structure that in train and val split the images are organized such as train/reception/00003724.jpg and val/raft/000050000.jpg

```python
sh download_data_pytorch.sh
```

### 2. Train the model
For train configs, please refer to conf/train.yaml 
```bash
python tools/train.py batch_size=128
```


### 3. carry out the inferring

Donwload pretrained weights from [[Google Drive]](https://drive.google.com/drive/folders/1NbV3NZlgbqnLSd9zwZoz8kFpNQjUYolT?usp=sharing), or use the weights you trained in local.
For infer configs, please refer to conf/infer.yaml 
```python
python tools/infer.py img_path=root/TorchScene/imgs/12.jpg weight_path=root/TorchScene/checkpoints/vision_transformer1654506720.0398195.ckpt
```

### 4. Convert a model to TorchScript

```python
python scripts/convert_torchscript.py
```

## Acknowledge

The dataset and basic code comes from [[MIT Place365]](https://github.com/CSAILVision/places365)

Thanks for the great work!
