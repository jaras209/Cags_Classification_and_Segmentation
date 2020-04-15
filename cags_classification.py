#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cags_dataset import CAGS
import efficient_net

'''
The goal of this assignment is to use pretrained EfficientNet-B0 model to achieve best accuracy in CAGS classification.

The CAGS dataset consists of images of cats and dogs of size 224×224224×224, each classified in one of the 34 breeds 
and each containing a mask indicating the presence of the animal. To load the dataset, use the cags_dataset.py module. 
The dataset is stored in a TFRecord file and each element is encoded as a tf.train.Example. Therefore the dataset is 
loaded using tf.data API and each entry can be decoded using .map(CAGS.parse) call.

To load the EfficientNet-B0, use the the provided efficient_net.py module. Its method 
pretrained_efficientnet_b0(include_top):

 - downloads the pretrained weights if they are not found;
 - it returns a tf.keras.Model processing image of shape (224, 224, 3)(224,224,3) with float values in range [0, 1][0,1] 
   and producing a list of results:
        - the first value is the final network output:
            - if include_top == True, the network will include the final classification layer and produce a distribution 
              on 1000 classes (whose names are in imagenet_classes.py);
            - if include_top == False, the network will return image features (the result of the last global average 
              pooling);
        - the rest of outputs are the intermediate results of the network just before a convolution with 
          \textit{stride} > 1stride>1 is performed.
          
An example performing classification of given images is available in image_classification.py.

A note on finetuning: each tf.keras.layers.Layer has a mutable trainable property indicating whether its variables 
should be updated – however, after changing it, you need to call .compile again (or otherwise make sure the list of 
trainable variables for the optimizer is updated). Furthermore, training argument passed to the invocation call decides 
whether the layer is executed in training regime (neurons gets dropped in dropout, batch normalization computes 
estimates on the batch) or in inference regime. There is one exception though – if trainable == False on a batch
normalization layer, it runs in the inference regime even when training == True.

This is an open-data task, where you submit only the test set labels together with the training script 
(which will not be executed, it will be only used to understand the approach you took, and to indicate teams). 
Explicitly, submit exactly one .txt file and at least one .py file.

The task is also a competition. Everyone who submits a solution which achieves at least 90% test set accuracy will get 
6 points; the rest 5 points will be distributed depending on relative ordering of your solutions.

You may want to start with the cags_classification.py template which generates the test set annotation in the 
required format.
'''


# Simple training data augmentation
def train_augment(image, label):
    # Horizontal flip with probability 0.5
    image = tf.image.random_flip_left_right(image)

    # Zooming of the image needs to be done this way
    image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
    image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
                                    tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
    image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])

    # Adjust the hue (odstín) of RGB images by a random factor.
    image = tf.image.random_hue(image, 0.08)
    # Adjust the saturation of RGB images by a random factor.
    image = tf.image.random_saturation(image, 0.6, 1.6)
    # Adjust the brightness of images by a random factor.
    image = tf.image.random_brightness(image, 0.05)
    # Adjust the contrast of an image or images by a random factor.
    image = tf.image.random_contrast(image, 0.7, 1.3)

    return image, label


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--tuning_epochs", default=400, type=int, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.4, type=int, help="Drop out rate.")
    parser.add_argument("--model", default="model_3.h5", type=str, help="Output model path.")
    parser.add_argument("--finetuned_model", default="model_3_finetuned.h5", type=str, help="Output model path.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # ===================== LOAD AND PREPARE DATA =======================================
    # Load the data
    cags = CAGS()

    # Load data in the form of TFRecord
    train_dataset = tf.data.TFRecordDataset("cags.train.tfrecord")
    dev_dataset = tf.data.TFRecordDataset("cags.dev.tfrecord")
    test_dataset = tf.data.TFRecordDataset("cags.test.tfrecord")

    # Decode data using CAGS.parse function on each element which turns each element into dictionary with:
    #   - "image"
    #   - "mask"
    #   - "label"
    # keys
    train_dataset = train_dataset.map(CAGS.parse)
    dev_dataset = dev_dataset.map(CAGS.parse)
    test_dataset = test_dataset.map(CAGS.parse)

    # For training and testing we need to have data in the form of pairs (image, label)
    train_dataset = train_dataset.map(lambda example: (example["image"], example["label"]))
    dev_dataset = dev_dataset.map(lambda example: (example["image"], example["label"]))
    test_dataset = test_dataset.map(lambda example: (example["image"], example["label"]))

    '''
    Another way to do this is to get the individual data to our hand directly by using numpy array:
        images = np.array(list(train_dataset.map(lambda example: example["image"])))
        labels = np.array(list(train_dataset.map(lambda example: example["label"])))
    and work with these arrays the same way as in the previous tasks
    '''

    # Prepare pipeline for training data:
    # - call `.shuffle(seed=args.seed)` to shuffle the data using
    #       the given seed and a buffer of the size of the whole data
    # - call `.map(train_augment)` to perform the dataset augmentation
    # - finally call `.batch(args.batch_size)` to generate batches
    train_data_pipeline = train_dataset.shuffle(5000, seed=args.seed)
    train_data_pipeline = train_data_pipeline.map(train_augment)
    train_data_pipeline = train_data_pipeline.batch(args.batch_size)

    # Prepare the pipeline for validation and test data
    # - just use `.batch(args.batch_size)` to generate batches
    dev_data_pipeline = dev_dataset.batch(args.batch_size)
    test_data_pipeline = test_dataset.batch(args.batch_size)

    # ================== CREATE AND TRAIN NEURAL NETWORK ==============================
    # Load the EfficientNet-B0 model without top classification layer as the convolutional base.
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

    # Freeze the convolutional base which prevents the weights from being updated during training
    efficientnet_b0.trainable = False

    # Create the model using Functional API
    inputs = tf.keras.layers.Input([CAGS.H, CAGS.W, CAGS.C])
    hidden = efficientnet_b0(inputs)[0]
    hidden = tf.keras.layers.Dropout(rate=0.2)(hidden)
    hidden = tf.keras.layers.Dense(500, activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(rate=0.3)(hidden)
    hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(rate=0.4)(hidden)
    hidden = tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=hidden)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    # Print model summary
    model.summary()

    # Train the model
    model.fit(x=train_data_pipeline, epochs=args.epochs,
              validation_data=dev_data_pipeline,
              callbacks=[tb_callback]
              )

    # Save the model
    model.save(args.model)

    # ============================= FINETUNING MODEL ================================================

    # Load model, COMMENT IT AFTER
    # model = tf.keras.models.load_model("model_100_epochs_2_dense_layers.h5", custom_objects={'swish': tf.nn.swish})

    # Unfreeze the convolutional base
    efficientnet_b0.trainable = True

    # Compile the model to make sure that the list of trainable variables for the optimizer is updated.
    # Usually a smaller learning rate is necessary, because the original model probably finished training with a very
    # small learning rate. A good starting point is one tenth of the original starting learning rate
    # (therefore, 0.0001 for Adam).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    # Print model summary
    model.summary()

    # Train the model
    model.fit(x=train_data_pipeline, epochs=args.tuning_epochs, initial_epoch=args.epochs,
              validation_data=dev_data_pipeline,
              callbacks=[tb_callback]
              )

    # Save the model
    model.save(args.finetuned_model)

    # ==================== EVALUATE THE MODEL ON TESTING DATA =================================

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
        # Predict the probabilities on the test set
        test_probabilities = model.predict(test_data_pipeline)
        for probs in test_probabilities:
            print(np.argmax(probs), file=out_file)
