#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import cags_segmentation_eval

from cags_dataset import CAGS
import efficient_net

'''
======================================== cags_segmentation ============================================
The goal of this assignment is to use pretrained EfficientNet-B0 model to achieve best image segmentation IoU score 
on the CAGS dataset. The dataset and the EfficientNet-B0 is described in the cags_classification assignment.

This is an open-data task, where you submit only the test set masks together with the training script (which will not 
be executed, it will be only used to understand the approach you took, and to indicate teams). Explicitly, submit 
exactly one .txt file and at least one .py file.

A mask is evaluated using intersection over union (IoU) metric, which is the intersection of the gold and predicted 
mask divided by their union, and the whole test set score is the average of its masks' IoU. A TensorFlow compatible 
metric is implemented by the class CAGSMaskIoU of the cags_segmentation_eval.py module, which can further be used to 
evaluate a file with predicted masks.

The task is also a competition. Everyone who submits a solution which achieves at least 85% test set IoU will get 6 
points; the rest 5 points will be distributed depending on relative ordering of your solutions.

You may want to start with the cags_segmentation.py template, which generates the test set annotation in the required 
format – each mask should be encoded on a single line as a space separated sequence of integers indicating the length 
of alternating runs of zeros and ones.
'''


# Simple training data augmentation.
# Every augmentation must be performed on image and mask as well.
def train_augment(image, mask):
    # Horizontal flip with probability 0.5
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Zooming of the image needs to be done this way
    '''
    image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
    image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
                                    tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
    image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
    '''

    # Adjust the hue (odstín) of RGB images by a random factor.
    image = tf.image.random_hue(image, 0.08)
    # Adjust the saturation of RGB images by a random factor.
    image = tf.image.random_saturation(image, 0.6, 1.6)
    # Adjust the brightness of images by a random factor.
    image = tf.image.random_brightness(image, 0.05)
    # Adjust the contrast of an image or images by a random factor.
    image = tf.image.random_contrast(image, 0.7, 1.3)

    return image, mask


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model", default="model_segment_3_finetuned.h5")
    parser.add_argument("--model", default="model_segment_3.h5", type=str, help="Output model path.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--tuning_epochs", default=60, type=int)
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

    # For training and testing we need to have data in the form of pairs (image, mask)
    train_dataset = train_dataset.map(lambda example: (example["image"], example["mask"]))
    dev_dataset = dev_dataset.map(lambda example: (example["image"], example["mask"]))
    test_dataset = test_dataset.map(lambda example: (example["image"], example["mask"]))

    '''
    Another way to do this is to get the individual data to our hand directly by using numpy array:
        images = np.array(list(train_dataset.map(lambda example: example["image"])))
        masks = np.array(list(train_dataset.map(lambda example: example["mask"])))
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

    # Create the model using Functional API:
    inputs = tf.keras.layers.Input([CAGS.H, CAGS.W, CAGS.C])
    # List of 6 outputs:
    # - the first value is the final efficientnet output
    # - the rest of outputs are the intermediate results of the network just before a convolution with stride > 1 is
    #   performed
    # - I.e. features[i]:
    #   features[0]: shape=(None, 1280)
    #           [1]: shape=(None, 7, 7, 1280)
    #           [2]: shape=(None, 14, 14, 12)
    #           [3]: shape=(None, 28, 28, 40)
    #           [4]: shape=(None, 56, 56, 24)
    #           [5]: shape=(None, 112, 112, 16)
    features = efficientnet_b0(inputs)

    # Create pyramid recreating image of shape (224, 224, 1) using extracted features from efficient net:
    # (None, 7, 7, 1280) -> (None, 7, 7, 256)
    x = features[1]
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # (None, 7, 7, 256) -> (None, 112, 112, 256) by parts: (None, k, k, 256) -> (None, 2k, 2k, 256)
    for feature in features[2:]:
        x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        f = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(feature)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.ReLU()(f)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        x = tf.keras.layers.Add()([x, f])

    # (None, 112, 112, 256) -> (None, 224, 224, 2) which is the predicted mask.
    outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same',
                                              activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), cags_segmentation_eval.CAGSMaskIoU(name="iou")],
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
    print("FINETUNING")
    # Load model, COMMENT IT AFTER
    # model = tf.keras.models.load_model(args.model, custom_objects={'swish': tf.nn.swish, 'CAGSMaskIoU': cags_segmentation_eval.CAGSMaskIoU()})

    # Unfreeze the convolutional base
    efficientnet_b0.trainable = True

    # Compile the model to make sure that the list of trainable variables for the optimizer is updated.
    # Usually a smaller learning rate is necessary, because the original model probably finished training with a very
    # small learning rate. A good starting point is one tenth of the original starting learning rate
    # (therefore, 0.0001 for Adam).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), cags_segmentation_eval.CAGSMaskIoU(name="iou")],
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
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as out_file:
        test_masks = model.predict(test_data_pipeline)
        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=out_file)