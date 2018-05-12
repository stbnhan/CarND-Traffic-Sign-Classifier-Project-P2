# **Traffic Sign Recognition** 

## CarND - Project 2

### Steven Han

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./reference/Train_Dataset_Sign_Counts.png "Train Dataset Count"
[image2]: ./reference/Test_Dataset_Sign_Counts.png "Test Dataset Count"
[image3]: ./reference/Valid_Dataset_Sign_Counts.png "Validation Dataset Count"
[image4]: ./reference/Test_Accuracy.png "Test Accuracy"
[image5]: ./reference/Validation_Accuracy.png "Validation Accuracy"
[image6]: ./reference/example_original.png "Original Images"
[image7]: ./reference/example_gray.png "Grayscale Images"

[image10]: ./reference/image1.png "Test Image 1"
[image11]: ./reference/image2.png "Test Image 2"
[image12]: ./reference/image3.png "Test Image 3"
[image13]: ./reference/image4.png "Test Image 4"
[image14]: ./reference/image5.png "Test Image 5"
[image15]: ./reference/image6.png "Test Image 6"
[image16]: ./reference/image7.png "Test Image 7"
[image17]: ./reference/image8.png "Test Image 8"
[image18]: ./reference/image9.png "Test Image 9"
[image19]: ./reference/image10.png "Test Image 10"

---
### Writeup / README

#### 1. Source Code

Link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale for the following reasons:  
1. Not necessary for edge/pattern detection
2. Keeping the model simpler by working with one less dimension
3. Faster computing speed

Then, I normalized the image data for consistency in all images (e.g. brightness)

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image6]
![alt text][image7]

I did not generate additional data because I had high enough accuracy and augmented data resulted in underfitting model.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	       						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 5x5     	| 1x1 stride, valid padding, output: 28x28x6	|
| RELU					| Activation									|
| Max pooling			| 2x2 stride, valid padding, output: 14x14x6 	|
| Convolution 5x5		| 1x1 stride, valid padding, output: 10x10x16	|
| RELU					| Activation   									|
| Max pooling			| 2x2 stride, valid padding, output: 5x5x16 	|
| Flatten				| output: 400									|
| Fully Connected		| output: 120									|
| RELU					| Activation   									|
| Dropout				| Probability: 0.5								|
| Fully Connected		| output: 84									|
| RELU					| Activation   									|
| Dropout				| Probability: 0.5								|
| Fully Connected		| output: 43									|
| Softmax				| softmax cross entropy							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimization Algorithm for the following reasons:  
1. Conveniently available in Tensorflow library
2. Computationally efficient
3. Easy to work with
  
Variable used for my model:
| Variable					|     Value				| 
|:-------------------------:|:---------------------:| 
| Epoch		       			| 30					| 
| Batch Size       			| 128					| 
| Learning Rate    			| 0.001					| 
| mu (truncated_normal)  	| 0						| 
| sigma (truncated_normal)  | 0.1					| 
| Dropout Probability		| 0.5					| 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.950
* test set accuracy of 0.997

![alt text][image4]
![alt text][image5]

If an iterative approach was chosen:  
* What was the first architecture that was tried and why was it chosen?  
	The LeNet architecture was used at first because I did an exercise training data using this architecture in previous lessons.  
* What were some problems with the initial architecture?  
	It was surprisingly good enough to work with as its accuracy was nearing at 90% on average. However, the default layers were not sufficient enough to get the accuracy I wanted.  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  
	First, I started off with basics: preprocessing. Changing RGB to Grayscale and adding a normalization to preprocessing step were known to add consistency. Then, I added a few more layers: dropout layer. This is also a known method to fix over-fit model by introducing generalization.  
* Which parameters were tuned? How were they adjusted and why?  
	Epoch and learning rate values were chosen with trial and error as it needs enough iteration to bring the model's accuracy to its full potential.  
	Other values were chosen by researching their default popular choice.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
	CovNets are very well known to be successful in analyzing visual imagery. 
	Dropout (as mentioned above) introduces a solution to prevent overfitting. It reduces the number of neurons by generalization which tend to result in learning more robust features.
	In addition, max pooling is similar to dropout layer.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image6]

Some of these images might be difficult to classify because they are pixelated and dark. They seem difficult even for human eyes to identify.
In addition, there are background edges and signs that might be distracting for the CNN to classify accurately.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			| Prediction	        						| Accuracy		|
|:-------------:|:---------------------------------------------:|:-------------:|
| Image 1		| Priority road									| 1.000			|
| Image 2		| Right-of-way at the next intersection			| 1.000			|
| Image 3		| Roundabout mandatory							| 1.000			|
| Image 4		| Bumpy road									| 1.000			|
| Image 5		| Speed limit (30km/h)							| 1.000			|
| Image 6		| Yield											| 1.000			|
| Image 7		| Speed limit (120km/h)							| 1.000			|
| Image 8		| Yield											| 1.000			|
| Image 9		| Road work										| 1.000			|
| Image 10		| Speed limit (70km/h)							| 1.000			|


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Test Result:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
