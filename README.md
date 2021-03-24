# Cags classification and segmentation
Task from [Deep Learning – Summer 2019/20](https://ufal.mff.cuni.cz/courses/npfl114/1920-summer#home). Part of this code was provided by 
[Milan Straka](https://ufal.mff.cuni.cz/milan-straka).

## cags_classification
The goal of this assignment is to use pretrained `EfficientNet-B0` model to achieve best accuracy in CAGS classification.

The [CAGS dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/demos/cags_train.html) consists of images of cats and dogs of size 224×224, each classified in one of the 34 breeds 
and each containing a mask indicating the presence of the animal. To load the dataset, use the [cags_dataset.py](https://github.com/jaras209/Cags_segmentation/blob/master/cags_segmentation.py) module. 
The dataset is stored in a [TFRecord](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) 
file and each element is encoded as a [tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example). Therefore the dataset is 
loaded using `tf.data` API and each entry can be decoded using `.map(CAGS.parse)` call.

To load the EfficientNet-B0, we use the the provided [efficient_net.py](https://github.com/jaras209/Cags_segmentation/blob/master/efficient_net.py) module. 

The model is build, trained and tested in [cags_classification.py](https://github.com/jaras209/Cags_segmentation/blob/master/cags_classification.py).
          
An example performing classification of given images is available in [image_classification.py](https://github.com/jaras209/Cags_segmentation/blob/master/image_classification.py).

## cags_segmentation

The goal of this project is to use pretrained `EfficientNet-B0` model to achieve best image segmentation IoU score 
on the CAGS dataset.

A mask is evaluated using `intersection over union (IoU)` metric, which is the intersection of the gold and predicted 
mask divided by their union, and the whole test set score is the average of its masks' IoU. A TensorFlow compatible 
metric is implemented by the class `CAGSMaskIoU` of the [cags_segmentation_eval.py](https://github.com/jaras209/Cags_segmentation/blob/master/cags_segmentation_eval.py) module, which can further be used to 
evaluate a file with predicted masks.

The model is build, trained and tested in [cags_segmentation.py](https://github.com/jaras209/Cags_segmentation/blob/master/cags_segmentation.py).
