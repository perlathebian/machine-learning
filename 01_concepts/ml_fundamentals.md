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

A trained model is evaluated to determine how well it learned. When evaluating, the model is given a labeled dataset, but only the dataset's features are given. Then the model's predictions are compared to the labels' true values.

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

#### R² (R-squared)

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
