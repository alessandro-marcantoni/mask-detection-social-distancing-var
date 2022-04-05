# __Mask Detection & Social Distancing__

> _Visione Artificiale e Riconoscimento_

This repository contains the project developed for the above course.
The two main parts are:  

* __Mask Detection__: this solution detects people's faces in pictures, ideally taken in a small room, and determines whether they are wearing a face mask correctly, not correctly or not at all.
* __Social Distancing__: this solution detects people in pictures, ideally taken in a small room, and checks whether they are abiding by the social distancing rules or not.

## __Requirements__

In order to use the solutions above, you will need to have ```python >= 3.7.12``` and ```pip >= 22.0.3``` installed on your machine. The required libraries can be easily downloaded and installed by running the following command: 
```
pip install -r requirements.txt
```

## __Mask Detection__

Inside the ```mask-detection``` directory you will find the following sources:  

* ```training.ipynb```: this is the notebook used to perform the training of the classifiers for the _mask classification_ task.
* ```performance_eval.ipynb```: this is the notebook used to test the performances of the entire solution.
* ```facemask_detection.py```: this is the script that performs the inference using both the ```retinaface``` model and the trained classifier; it takes two command line parameters:
    
    * The path of the video to be inferred or "```webcam```" to infer the webcam video stream in real time.
    * Either "```ssd```" or "```svm```" depending on which classifier you want to use.
    ```
    python facemask_detection.py <path/to/video> <model>
    ```

### __Resources__

You will need to download and unzip the [```facemask-detection-resources.zip```](https://drive.google.com/file/d/1-Aa3MxphxxxyAvK0pRfmk8ETEJQbA9nD/view?usp=sharing) archive in the ```mask-detection``` folder. It contains:

* The ```retinaface``` model for face detection;
* The ```mobilenet``` model for mask classification;
* The ```SVC``` model for mask classification;
* The dataset used for training;
* The video and labels used for test.

## __Social Distancing__

Inside the ```social-distancing``` directory you will find the following sources:

* ```social_distancing_eval.ipynb```: this is the notebook used to test the performances of different neural networks and choose the best one.
* ```social_distancing_matrix.py```: this is the script used to generate the transformation perspective matrix which is fundamental for the social distancing verification task.
* ```social_distancing_verification.py```: this is the script that performs the inference using the best model; it takes one command line parameter:
    
    * The path of the video to be inferred.
    ```
    python social_distancing_verification.py <path/to/video>
    ```

### __Resources__

You will need to download and unzip the [```social-distancing-resources.zip```](https://drive.google.com/file/d/1cvSxiu1t7jY7KLyjDlj0Z5HQAqanle6f/view?usp=sharing) archive in the ```social-distancing``` folder. It contains:

* Video example of people in a room to try the inference script;
* Images and labels used for test;
* Transformation matrix for both the video example and test set. 
