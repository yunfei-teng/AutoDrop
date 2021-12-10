# AutoDrop: Training Deep Learning Models with Automatic Learning Rate Drop
We provide the source codes used to generate the experimental results in our paper.

## Reproducibility
To reproduce the results, simply run the bash files:
  * For ResNet model and CIFAR data set: `bash resnet_cifar.sh`
  * For WRN model and CIFAR data set: `bash wrn_cifar.sh`
  * For ResNet model and ImageNet data set: `bash resnet_imagenet.sh`

For ImageNet experiment, please modify the paths to train data and validation data in `data.py`.

## Results
The result of each experiment will be saved in `results` folder by its experiment name. One could check the saved `.pt` file to print out the necessary results.
