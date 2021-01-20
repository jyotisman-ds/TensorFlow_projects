# TensorFlow Projects

- [General Overview](#general-overview)
- [List of Projects](#list-of-projects)
  * [Beans Classifier](#beans-classifier)
  * [Dog Breed Classifier](#dog-breed-classifier)
- [Credits](#credits)

## General Overview
This repo is a compilation of some of the deep learning projects that I worked on independently or as part of some training. This is by no means an exhaustive list. Its mostly for demonstration of the skills that I have acquired over the years as a learner and an enthusiast.

## List of Projects

### Beans Classifier

**Notebook** : [beans_classifier](https://github.com/jyotisman-ds/TensorFlow_projects/blob/main/Beans_Classifier/Beans_fullCalssifier.ipynb)

#### Model Demo
This model learns the Beans images [dataset](https://www.tensorflow.org/datasets/catalog/beans) from tensorflow datasets. The images are first processed through an augmentation pipeline before being fed into  a tf.data pipeline.

![Browser Model](/images/PredictingBeans.png)

#### Overview
The trained model is deployed on to the browser using tensorflow.js. It can take a picture of a bean leaf and predict whether it belongs to one of the two diseased categories - bean_rust or angular leaf spot or if its healthy.  

#### Technical Aspects
- The model was trained in Google Colab with GPU settings. The dataset is already sub-divided into train, validation and test datasets and data loading is extremely simplified with the tensorflow_datatsets module.
- A deep learning model is implemented with a Keras Sequential layer consisting of Convolutional, Maxpooling and Dense Layers.
- Random augmentations like flipping, rotation and zoom were implemented. This provided more variety to the training data while also preventing overfitting.
- A Dropout layer is also used to further prevent overfitting.
- A proper tf.data pipeline is set to shuffle, batch and prefetch data for the training process.
- Deployment is done through Tensorflow.js services which enables locally hosting our model on the browser simply by converting our trained Keras model into json model with the following
```python
saved_model_path = "./{}.h5".format(int(time.time()))
model.save(saved_model_path)
!tensorflowjs_converter --input_format=keras {saved_model_path} ./
```
_Tools : Python, Tensorflow, Keras, Tensorflow_datasets, Tensorflow.js, html, javascript, Matplotlib, Google Colab_

### Dog Breed Classifier

**Notebook** : [dog_breed_classifier](https://github.com/jyotisman-ds/TensorFlow_projects/blob/main/Dog_breed_classifier/Dog_breed_classifier_optimized.ipynb)

#### Model Demo
This model learns the dog images [dataset](https://www.tensorflow.org/datasets/catalog/stanford_dogs) from tensorflow datasets. The images are first processed through an augmentation pipeline before being fed into  a tf.data pipeline. Since this is a complicated dataset, we use transfer learning to train our model here. I use the InceptionV3 model with the default 'imagenet' weights. Inception is a highly optimized model for image classification benchmarked against the famous imagenet dataset. I set the last few layers of the model to be trainable for my purposes.

![Browser Model](/images/dog_breed.png)

_Image credits for the picture used in the classification above :_ https://dogtime.com/dog-breeds/pekingese#/slide/1

#### Overview
As with the beans classifier, we deploy the model onto the local browser using tensorflow.js. It consists of 120 classes of dog breeds with every breed consisting of around 150 images on an average. We have 12000 training images and we further divide the provided test dataset into a validation and a test dataset.

#### Technical Aspects
- This is more or less the same as the beans classifier.
- The additional thing that I do here is to interleave the training dataset to optimize the data reading/extraction process.
- An important aspect of the training is also to use the bounding boxes that is provided with the tensorflow dataset. So the image is first cropped to the bounding box dimensions provided and then later resized to the common dimension of (240, 240, 3) for training (This is what the browser model - model.json is trained with even though the notebook shows (224,224,3). Colab runs into memory issues for a larger image size).
- The classification_report module from the sklearn library is also used to look at some of the other metrics like Recall and precision. Even though this problem does not really require investigating false negatives or false positives, its still an interesting statistical summary to look through given the huge number of labels.

```
      precision   recall   f1-score   support

0       0.86      0.67      0.75        27
1       0.92      0.92      0.92        39
2       0.91      0.89      0.90        80
3       0.93      0.81      0.86        31
4       0.88      0.88      0.88        49
5       0.90      0.94      0.92        49
6       0.95      0.93      0.94        45
7       0.84      0.91      0.88        35
8       0.89      0.71      0.79        34
                        .
                        .
                        .
```

- Finally, a confusion matrix is plotted to quick visual summary of how the classifier does on unseen data. The brightly lit diagonal justifies the ~85% accuracy seen before.  

_Tools : Python, Tensorflow, Keras, sklearn, Tensorflow_datasets, Tensorflow.js, html, javascript, Matplotlib, Google Colab_

## Credits
A huge shoutout to the Deep Learning coursera community especially their [Deep Learning](https://www.coursera.org/specializations/deep-learning) and Tensorflow [training](https://www.coursera.org/professional-certificates/tensorflow-in-practice) and [deployment](https://www.coursera.org/specializations/tensorflow-data-and-deployment) specialization courses.
