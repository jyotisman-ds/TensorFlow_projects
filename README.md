# TensorFlow Projects

- [General Overview](#general-overview)
- [List of Projects](#list-of-projects)
  * [Beans Classifier](#beans-classifier)
  * [Dog Breed Classifier](#dog-breed-classifier)
  * [Sentiment Classifier](#sentiment-classifier)
- [Credits](#credits)

## General Overview
This repo is a compilation of some of the deep learning projects that I worked on independently or as part of some training. This is by no means an exhaustive list. Its mostly for demonstration of the skills that I have acquired over the years as a learner and an enthusiast. Anyone interested in trying out these models in their local browser (apart from Sentiment Classifier) can do the following :

1. Download the model folder you want to try.
2. (only for Google Chrome)Add the [web server](https://chrome.google.com/webstore/detail/web-server-for-chrome/ofhbbkphhbklhfoeikjpcbhemlocgigb?hl=en) for Chrome extension to Google Chrome.
3. Point to the downloaded model folder from step 1 in the web server app and click on the web server URL link.
4. Click on the html file.
5. Finally, remember to activate the developer tools option from the chrome menu to look at the console logs if interested.

Now, you can upload your own images and have fun!

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
This model learns the [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs) dataset from tensorflow datasets. The images are first processed through an augmentation pipeline before being fed into  a tf.data pipeline. Since this is a complicated dataset, we use transfer learning to train our model here. I use the InceptionV3 model with the default 'imagenet' weights. Inception is a highly optimized model for image classification benchmarked against the famous imagenet dataset. I set the last few layers of the model to be trainable for my purposes.

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

- Finally, a confusion matrix is plotted to get a quick visual summary of how the classifier does on unseen data. The brightly lit diagonal justifies the ~85% accuracy seen before.  

_Tools : Python, Tensorflow, Keras, sklearn, Tensorflow_datasets, Tensorflow.js, html, javascript, Matplotlib, Google Colab_

### Sentiment Classifier

**Notebook** : [sentiment_classifier](https://github.com/jyotisman-ds/TensorFlow_projects/blob/main/Sentiment_Classifier/sentiment_classifier.ipynb)

#### Model Demo

This model learns the [yelp_polarity_reviews](https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews) dataset from tensorflow datasets. This is a simple text classifier with 1D convolutional and global average pooling layers.

#### Overview

The dataset is significantly huge with close to 600,000 training and around 40,000 test examples. As usual, I split the test set into a validation and test set. It's a binary classifier with review polarities labelled as a negative '0' label and a positive '1' label.

#### Technical Aspects

- Data preprocessing included removing common 'stopwords' from all the splits. The list shown below is taken from this [website](http://mlg.ucd.ie/datasets/bbc.html). Other common preprocessing steps for text datasets were taken care of by the TextVectorization layer which limits the max_length of the reviews and also pads them whenever necessary.

```python

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

```

- The TextVectorization layer more importantly also creates a Tokenizer of 10,000 most common words from the corpus in the training dataset. This layer also indexes these tokens which can then be used to vectorize the texts in all the splits. This layer can be used just as separate preprocessing step or also as a layer directly for the model given that it is inherited from the keras.layers class.

- For word embeddings, we create an embedding layer as part of the model itself with 64 dimension embeddings to be learned during training.  

- With a modest Conv1D and GlobalAveragePooling1D() layer, an accuracy of ~ 94% is achieved. One can further increase this with the inclusion of a TF-IDF vectorizer which accounts for the relative importance of words in the corpus. Or one can also use some pre-trained word embeddings like [Glove](https://nlp.stanford.edu/projects/glove/). Other common preprocessing steps include stemming (focusing on the root words by cutting off their stems - e.g. study(-ing), like(-ly)), filtering of words based on their frequency of occurrence, etc. 

- A straightforward next step would be to play with sequence layers cause after all texts are meaningful only if they have a sequential structure. One can add a few LSTM or GRU layers to check that. It probably wouldn't matter a lot for this task where a few positive or negative words will find a reliable mapping to the label. But for tasks like text generation, language translation where sequence learning is a core task, it's a must.    

## Credits
A huge shoutout to the Deep Learning coursera community especially their [Deep Learning](https://www.coursera.org/specializations/deep-learning) and Tensorflow [training](https://www.coursera.org/professional-certificates/tensorflow-in-practice) and [deployment](https://www.coursera.org/specializations/tensorflow-data-and-deployment) specialization courses.
