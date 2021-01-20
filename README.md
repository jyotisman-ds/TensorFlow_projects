# TensorFlow Projects

## Table of Content
 * [General Overview](#overview)
 * [List of Projects](#projects)
 * [Credits](#credits)

## General Overview
This repo is a compilation of some of the deep learning projects that I worked on independently or as part of some training. This is by no means an exhaustive list. Its mostly for demonstration of the skills that I have acquired over the years as a learner and an enthusiast.

## List of Projects

### [Beans Classifier](https://github.com/jyotisman-ds/TensorFlow_projects/blob/main/Beans_Classifier/Beans_fullCalssifier.ipynb)

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

## Credits
A huge shoutout to the Deep Learning coursera community especially their [Deep Learning](https://www.coursera.org/specializations/deep-learning) and Tensorflow [training](https://www.coursera.org/professional-certificates/tensorflow-in-practice) and [deployment](https://www.coursera.org/specializations/tensorflow-data-and-deployment) specialization courses.
