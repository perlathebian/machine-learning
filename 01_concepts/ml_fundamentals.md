# ML Fundamentals - Learning Notes

_Source: Google Intro to ML + Google ML Crash Course + Scikit-learn Docs_

---

## What is Machine Learning?

### Definition

Machine learning is a system that gradually learns how to make useful predictions by studying lots of data to discover connections and correlations.

### What is an ML model?

A model is a mathematical relationship derived from data that an ML system uses to make predictions.

### Types of ML Systems

Based on how they learn to generate content or make predictions, ML systems fall into one or more categories:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning
4. Generative AI

---

## Supervised Learning

Supervised Learning can make predictions after seeing lots of data with correct answers then discovering the connections between the elements in the data that produce the correct answers. (Supervised in the sense that a human gives the system data with known correct results.)

Common use cases for supervised learning are classification and regression.

### Classification

A classification model predicts the likelihood that something belongs to a category. It outputs a value that states whether or not something belongs to a particular category.
Example: whether an email is spam or whether photo contains a cat

Classification models are divided into two groups:

- Binary Classification Models: output a value from a class that contains only two values (e.g., a model that outputs 'rain' or 'no rain')
- Multiclass Classification Models:output a vlaue form a class that contains more than two values (e.g., a model that can output either 'rain', 'hail', 'snow', or 'sleet')

### Regression

A regression model predicts a numeric value.
Example: a weather model that predicts the amount of rain is a regression model.

### Foundational Supervised Learning Concepts

- Data
- Model
- Training
- Evaluating
- Inference

### Data

Data is the driving force of ML. Data comes in the form of words and numbers stored in tables, or as the values of pixels and waveforms captured in images and audio files. Related data is stored in datasets (e.g., weather info, house pricing, ...).
Datasets are made up of individual examples that contain features and a label. Features are the values that a supervised model uses to predict the label. Examples that contain both features and a label are called labeled examples. In contrast, unlabeled examples contain features but no label. After the model is created, it predicts the label from the features.

A dataset is characterized by its size and diversity. Size indicates the number of examples. Diversity indicates the range the examples cover. Good datasets are both large and highly diverse.
A dataset can also be characterized by the number of its features.

### Model

In supervised learning, a model is the complex collection of numbers that define the mathematical relationship from specific input feature patterns to specific output label values. The model discovers these patterns through training.

### Training

To train a model to make predictions, we give it a dataset with labeled examples. The model's goal is to predict the best solution for predicting the labels from the features. The model finds the best solution by comparing the predicted value to the label's actual value. The difference between these two values is defined as the loss, and based on it the model gradually updates its solution. In other words, the model learns the mathematical relationship between the features and the label so that it can make the best predictions on unseen data. That gradual understanding is also why large and diverse datasets produce a better model.

### Evaluating

A trained model is evaluated to determine how well it learned. When evaluating, the model is given a labeled dataset, but only the dataset's features are given. Then the model's predictions are compared to the label's true values.

### Inference

Inferences are the predictions made by the trained, then evaluated model, on unlabeled examples.

---

## Unsupervised Learning

An unsupervised learning model aims to identify meaningful patterns in a dataset. One example of the techniques these models rely on is called clustering which involves organizing similar data into groups/clusters. Clustering differs from classification because categories aren't defined by the human.

---

## Reinforcement Learning

RL models make predictions by getting rewards or penalities based on actions performed within an environment. A reinforcement learning system generates a policy that defines the best strategy for getting the most rewards.
Example: training robots to perform tasks...

---

## Generative AI

GenAI is a class of models that creates content from user input.
Example: creating unique images, summarizing articles, explaining how to perform a task, or editing photos.

GenAI can take a variety of inputs and create a variety of outputs, like text, images, audio, and video. It can also take and create combinations of those.
A partial list of some inputs and outputs for generative models:

- Text-to-text
- Text-to-image
- Text-to-video
- Text-to-code
- Text-to-speech
- Image and text-to-image

### How does generative AI work?

At high-level, generative models learn patterns in data with the goal to produce new but similar data. To produce unique and creative outputs, they are intially trained using an unsupervised approach, where the mdoel learns to mimic the data it's trained on. The model is sometimes trained further using supervised or reinforcement learning on specific data related to tasks the model might be asked to perform.

---

## Linear Regression

Statistical technique used to find the relationship between variables. In ML context, linear regression finds the relationship between features and label.

### Linear Regression Equation

In algebraic terms, the model would be defined as $y = mx + b$, where

- $y$ is the value we want to predict
- $m$ is the slope of the line
- $x$ is our input value
- $b$ is the y-intercept

In ML, the equation for a linear regression model is written as: $y' = b + w_{1} x_{1}$, where

- $y'$ is the predicted label (output)
- $b$ is the bias of the model (same concept as y-intercept). In ML, bias is referred to sometimes as $w_{0}$. Bias is a parameter of the model and is calculated during training.
- $w_{1}$ is the weight of the feature (same concept as slope). It is also a parameter of the model and calculated during training.
- $x_{1}$ is a feature (input)

The above equation uses only one feature. A more sophisticated model might rely on multiple features, each having a separate weight. For example, a model relying on five features would be defined as follows: $y' = b + w_{1} x_{1} + w_{2} x_{2} + w_{3} x_{3} + w_{4} x_{4} + w_{5} x_{5}$

### Loss

Loss is a numeric metric that describes how wrong a model's predictions are. It measures the distance between the model's predictions and the label's actual values. The goal of traning is to minimize the loss to its lowest possible value. Since we only care about the distance, we remove the sign either by taking the absolute value of the difference, or by squaring it.

#### Types of Losses in Linear Regression

1. $\mathrm{L}_1$ Loss: sum of the absolute values of the difference between teh actual values and the predicted values
   $$
   \sum_{i=1}^{N} \lvert \text{actual value}_i - \text{predicted value}_i \rvert
   $$
2. Mean Absolute Error (MAE): the average of $\mathrm{L}_1$ losses over a set of $N$ examples
   $$
   \frac{1}{N} \sum_{i=1}^{N} \lvert \text{actual value}_i - \text{predicted value}_i \rvert
   $$
3. $\mathrm{L}_2$ Loss: sum of squared difference between predicted and actual values
   $$
   \sum_{i=1}^{N} (\text{actual value}_i - \text{predicted value}_i)^2
   $$
4. Mean Squared Error (MSE): the average of $\mathrm{L}_2$ losses across a set of $N$ examples
   $$
   \frac{1}{N} \sum_{i=1}^{N} (\text{actual value}_i - \text{predicted value}_i)^2
   $$
5. Root Mean Squared Error (RMSE): the square root of the mean squared error (MSE)
   $$
   \sqrt{
   \frac{1}{N} \sum_{i=1}^{N} (\text{actual value}_i - \text{predicted value}_i)^2
   }
   $$

MAE and RMSE give errors in the same units as what you're predicting, so humans can understand them more easily. MAE represents the average prediction error, whereas RMSE represents the spread of the errors.

#### Choosing a Loss

Most feature values typically fall within a distinct range. A value outside the typical range would be considered an outlier. An outlier can also refer to how far off a model's predictions are from the actual values. Considering how outliers need to be treated, the best loss function can be chosen.
Example:

- MSE moves the model more towards outliers, MAE doesnt.
- $\mathrm{L}_2$ loss incurs a much higher penalty for an outlier than $\mathrm{L}_1$ loss.

The relationship between the model and the data:

- MSE: The model is closer to the outliers but further away from most of the other data points.
- MAE: The model is further away from the outliers but closer to most of the other data points.

#### Gradient Descent

This is a mathematical technique that iteratively finds the weights and bias that produce the model with the lowest loss. Gradient descent finds the best weights and bias by repeating the following steps for a user-defined number of iterations. The model begins training with randomized weights and biases near zero, and then:

1. Calculate the loss with the current weight and bias
2. Determine the direction to move the weight and bias that reduce loss
3. Move the weight and bias a small amount in the direction that reduces loss
4. Back to step 1 and repeat until the model can't reduce loss any more

The point representing the minimum loss for the model, is typically greater than 0. A loss of 0 would mean that the model fits every data point exactly, which is usually a sign of overfitting (i.e., the model is too complex/powerful).

### Linear Regression: Hyperparameters

Hyperparameters are variables that control different aspects of training. Common hyperparameters are: learning rate, batch size, and epochs.
In contrast, parameters are variables like weights and bias that are part of the model itself. (i.e., hyperparameters are values you control, parameters are values calculated by the model during training).

#### Learning Rate

This is a floating point number that you set to influence how quickly a model converges. If the learning rate was too low, the model can take a long time to converge. If it was too high, the model never converges. The goal is to pick a learning rate that is not too low nor too high, so that the model converges quickly. This rate determines the magnitude of changes to apply to the weights and bias during each step of the gradient descent process. The model multiplies the gradient by the learning rate to determine the model's parameters for the next iteration (the ""small amount" mentioned in step 3 of the gradient descent process refers to the learning rate).

#### Batch Size

This refers to the number of examples the model processes before updating its weights and bias. Using a full batch isn't practical when a dataset contains hundred of thousands or millions of examples, and for that there are two techniques we can use to get the right gradient on average without needing to look at every example before updating parameters:

- Stochastic Gradient Descent (SGD): uses a single example (a batch size of one) in each iteration. Given enough iterations, SGD works but is very noisy. Noise refers to variations during training that cause the loss to increase during an iteration rather than decrease. (The word stochastic means that the example is chosen at random)

- Mini-batch Stochastic Gradient Descent (Mini-batch SGD): compromise between SGD and full batch. The batch size can be anywhere greater than 1 and less than the number of data points. The model chooses the examples in each batch randomly, averages their gradients, then updates its parameters once per iteration.

Note: Noise isn't always bad. A certain amount of noise can be a good thing (can help a model generalize better and find the optimal weights and bias in a neural network).

#### Epochs

During training, an epoch means that the model processed each example in the training set exactly once.
Example: given a training set with 1000 examples, and a mini-batch of 100 examples, it will take the model 10 iterations to complete one epoch (1 epoch = number of examples / batch size).
Training requires multiple epochs, meaning the model needs to process every example in the training set multiple times. This hyperparameter is set before training begins. In general, more epochs produces a better model, but takes more time to train.

---

## The ML Workflow

1.
2.
3.

---

## Train/Test Split

**Why we need it:**

**How it works:**

**Data leakage:**

---

## Feature Engineering

### Handling Missing Values

### Encoding Categorical Variables

### Feature Scaling

---

## Evaluation Metrics

### Classification Metrics

#### Accuracy

#### Precision

#### Recall

#### F1-Score

### Regression Metrics

#### MAE (Mean Absolute Error)

#### RMSE (Root Mean Squared Error)

#### R² (R-squared)wwww

---

## Overfitting vs Underfitting

### Overfitting

### Underfitting

### How to Detect

### How to Fix

---

## Model Selection

### When to Use What

---

## Key Takeaways
