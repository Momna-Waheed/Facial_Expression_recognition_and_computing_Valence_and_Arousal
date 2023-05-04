# Facial_Expression_recognition_and_computing_Valence_and_Arousal
Developement of Convolutional Neural Network (CNN) architecture for recog-
nizing facial expressions and computing valence and arousal for the
given dataset. 

<!--ts-->
Contents
<!--te-->

<!--ts-->
* [Goal](##Goal)
* [Dataset](##DataSet)
* [Methodology and Results](##MethodologyandResults)
	* [ResNet18](###ResNet18)	
	* [EfficinetNet](###EfficinetNet)
* [Model Testing and Enviornment setting](##ModelTestingandEnviornmentsetting)
<!--te-->

## Goal 
 In this project, the goal is to use appropriate Convolu-
tional Neural Network (CNN) architectures to recognize facial expressions and
compute valence and arousal for a given dataset of face images. The project
aims to use at least two CNN baselines, compare their results, and evaluate
their performance in terms of accuracy, F1 score, and other metrics.

## DataSet 
The dataset consists of facial images and their cor-
responding labels for six basic emotions (happiness, sadness, anger,
surprise, fear, and disgust), valence (degree of positivity or negativ-
ity of an emotion) and arousal (degree of intensity of an emotion).
The provided dataset consisted of facial images which were categorized into
8 facial expressions i.e., (0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5:
Disgust, 6: Anger, 7: Contempt).Valence is a continuous value ranging from
-1 to +1, except for the categories Uncertain and No-face, where the value is
-2, and Arousal is a continuous value ranging from -1 to +1, except for the
categories Uncertain and No-face, where the value is -2, with -1 representing
Tired and +1 representing Active.. The annotation files were provided in .npy
format and those files were used to formulate the labels against each image.The dataset link has not been attached but can be provided on demand.
The below figure shows some of the examples from training dataset.

![Capture](https://user-images.githubusercontent.com/59650991/236347420-e7452071-0657-45e1-9d18-4437c79e4166.PNG)

## Methodology and Results
To achieve mentioned goal, two CNN baseline architectures ResNet18 and EffeicientNet have been modified and implemented.
### Resnet18
ResNet-18, a light and simple architecture with a smaller number of layers, was
chosen as the base model for the project due to limited GPU access available on
Google Colab. The original architecture of the model consisted of and average
pooling layer followed by a fully connected layer which was modified by defining
a custom classifier having one fully connected layer having 8 output nodes to
perform classification and another fully connected layer consisting of 2 output
nodes to perform regression on valence and arousal values. For training this
model the training data was split into 80/20 ratio to be used as training and
validation set. The required parameters were initialized and instance of custom
classifier was initialized and was trained for 10 epochs. Checkpoints were saved
at each epoch and best model stats were also saved. It was observed that the
training and validation accuracy as well as loss improved for the few initial epochs and the model started over-fitting soon afterwards. Early stopping was
utilized and best model was saved. The best model was then tested against the
separately provided test set. The predicted outputs were utilized to compute
the performance measures of the model.It was observed that since the training
set had a huge class imbalance that’s why its 20% that was used as validation
set also had same issue thus the testing results came out to be poor. Since the
resources were limited and training took way longer thus model wasn’t trained
again however, it is expected that if the class imbalance issues of data set were
resolved and the model would have been retrained on it the performance would
have improved. The below figures show the results of model obtained on testing.

![resnetAcc](https://user-images.githubusercontent.com/59650991/236348040-257d6977-bd10-413c-8265-e0ad59651854.PNG)
![ResnetLoss](https://user-images.githubusercontent.com/59650991/236348066-20cc5772-9a5a-4c63-a001-8d766ad8ca04.PNG)
![resnetregg](https://user-images.githubusercontent.com/59650991/236348089-70ca7d2b-3883-48c2-94ca-f5a8a13e2495.PNG)
![Resnetclass](https://user-images.githubusercontent.com/59650991/236348110-44ae6633-1d07-4e5c-8947-770916b974e8.PNG)

### EfficinetNet
EfficientNet was chosen as the second base model due to its ability to efficiently
utilize limited computing resources, such as the limited GPU access available
on Google Colab, while still providing good performance on large datasets.Here
again, the model’s architecture was modified by defining a custom classifier
having one fully connected layer having 8 output nodes to perform classifica-
tion and another fully connected layer consisting of 2 output nodes to perform
regression on valence and arousal values. For training this model the training
data was split into 80/20 ratio to be used as training and testing data while
the seperatley provided dataset having balanced classes was used as a valida-
tion set to ensure proper learning by the model.The required parameters were
initialized and instance of custom classifier was initialized and was trained for 10 epochs initially but it was observed that there was a steady performance
improvement by the model so it was then trained for 10 more epochs mak-
ing 20 epochs in total. No overfitting was observed by the model even after
20 epochs and it is expected that further training improve the performance of
model. The below figures show the results of model obtained on testing.

![ENLoss](https://user-images.githubusercontent.com/59650991/236348302-86ab8d7f-fc86-4066-ad88-524e943a8883.PNG)
![ENAcc](https://user-images.githubusercontent.com/59650991/236348317-72fef089-560d-4f2d-8bfc-d3dc1f4b7d9d.PNG)
![ENClass](https://user-images.githubusercontent.com/59650991/236348341-f8976ea8-ffd8-454a-ab1b-3efcca93e123.PNG)
![ENReg](https://user-images.githubusercontent.com/59650991/236348348-ff95dd13-d493-49bf-879c-8c4e08fac363.PNG)


# Model Testing and Enviornment setting
Step by step guideline to execute the model has been provided in the attached notebook along with the all the dependancies and required libraries

