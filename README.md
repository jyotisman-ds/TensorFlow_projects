# TensorFlow Projects

## Table of Content
 * [List of Projects](#projects)


## List of Projects

- ### [Beans Classifier](https://github.com/jyotisman-ds/TensorFlow_projects/blob/main/Beans_fullCalssifier.ipynb)

    - Model Demo : This model learns the Beans images [dataset](https://www.tensorflow.org/datasets/catalog/beans) from tensorflow datasets. The images are first processed through an augmentation pipeline before being fed into  a tf.data pipeline.

    ![Browser Model](/images/PredictingBeans.png)

    - Overview : The trained model is deployed on to the browser using tensorflow.js. It can take a picture of a bean leaf and predict whether it belongs to one of the two diseased categories - bean_rust or angular leaf spot or id its healthy.  

    - Technical Aspects : The model was trained in Google Colab with GPU settings. The dataset is already sub-divided into train, validation and test datasets and data loading is as simple as the following
```python
import tensorflow_datasets as tfds
beans, info = tfds.load('beans', with_info=True, as_supervised=True)
train_ds = beans['train']
valid_ds = beans['validation']
test_ds = beans['test']
```
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
