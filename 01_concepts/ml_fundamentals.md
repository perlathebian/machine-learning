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

<details>
<summary>Show Details</summary>

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

#### Types of Losses in Linear Regression

1. **L₁ Loss**: sum of the absolute values of the difference between the actual values and the predicted values

$$
\sum_{i=1}^{N} | \text{actual value}_i - \text{predicted value}_i |
$$

2. **Mean Absolute Error (MAE)**: the average of L₁ losses over a set of N examples

$$
\frac{1}{N} \sum_{i=1}^{N} | \text{actual value}_i - \text{predicted value}_i |
$$

3. **L₂ Loss**: sum of squared difference between predicted and actual values

$$
\sum_{i=1}^{N} (\text{actual value}_i - \text{predicted value}_i)^2
$$

4. **Mean Squared Error (MSE)**: the average of L₂ losses across a set of N examples

$$
\frac{1}{N} \sum_{i=1}^{N} (\text{actual value}_i - \text{predicted value}_i)^2
$$

5. **Root Mean Squared Error (RMSE)**: the square root of the mean squared error (MSE)

$$
\sqrt{ \frac{1}{N} \sum_{i=1}^{N} (\text{actual value}_i - \text{predicted value}_i)^2 }
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

</details>

---

## Logistic Regression

<details>
<summary>Show Details</summary>

### What is Logistic Regression

It would be useful if a regression model predicted a formal probability, a value from 0 to 1, representing the chances that some condition happens. This type of regression task is called logistic regression. To get a linear model to output a probability, we need to transform the model to apply some limits so it outputs continuous values that fall within the range from 0 to 1. To find the mathematical function we can use to do this, we can think of two curves; an exponential curve and a hyperbola. Both produce outputs that approach a limit in at least one direction, but neither range is constrained to values between 0 and 1. If we combine the two formulas, we create a curve whose output is squished between 0 and 1. As the input decreases to negative infinity, the output approaches 0, adn as the input increases to infinity, the output approaches 1. This is a sigmoid curve, one of a family of s-shaped curves called logistic functions, from which logistic regression gets its name.
Logistic regression is a popular technique for building models that discriminate between two possible outcomes, including classification tasks.
Logistic regression model's output can be used as-is (probability estimate, e.g., 0.93 or 93%), or can be converted to a binary category (such as True or False, Spam or Not Spam).

### Sigmoid Function

This is the standard logistic function, and has formula: $f(x) = \frac{1}{1 + e^{-x}}$

#### Transforming Linear Output Using Sigmoid Function

The linear component of a logistic regression model has the following equation: $z = b + w_1 x_1 + w_2 x_2 + \dots + w_N x_N$, where

- $z$ is the output of the linear equation, also called log odds
- $b$ is the bias
- the $w$ values are the model's learned weights
- the $x$ values are the feature values

To obtain the logistic regression prediction, the $z$ value is passed to the sigmoid function resulting in a value between 0 and 1 (the probability): $y' = \frac{1}{1 + e^{-z}}$, where

- $y'$ is the output of the logistic regression model
- $e$ is the Euler number (~2.7)
- $z$ is the linear output

### Loss and Regularization

Logistic regression models are trained using the same process as linear regression, except that for logistic regression:

- models use log loss as the loss function instead of squared loss
- regularization is applied to prevent overfitting

#### Log Loss

Squared loss, $\text{L}_2$, works well for the linear model where the rate of change of output values is constant. In contrast, the rate of change of a logistic regression model is not constant. When the log odds ($z$) value is closer to 0, small increases in $z$ result in much larger changes in $y$ than if $z$ was a large positive or negative number.
The loss function for logistic regression is Log Loss. The Log Loss equation returns the logarithm of the magnitude of change, rather than just the distance from the data to prediction. Log Loss is calculated as follows:

$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(y'_i) + (1 - y_i)\log(1 - y'_i) \right]
$$

where,

- $N$ is the number of labeled examples in the dataset
- $i$ is the index of an example in the dataset (example ($\text{x}_3$, $\text{y}_3$) is the third example in the dataset)
- $\text{y}_i$ is the label for the ith example (must be 0 or 1)
- $\text{y}_i$' is your model's prediction for the ith example (between 0 and 1), given the set of features in $\text{x}_i$

#### Regularization

A mechanism for penalizing model complexity during training, extremely important in logistic regression modeling.
Two ways to decrease model complexity:

- $\text{L}_2$ regularization
- Early stopping: limiting the number of training steps to halt training while loss is decreasing

</details>

---

## Classification

<details>
<summary>Show Details</summary>
Classification is the task of predicting which of the set of classes (cetagories) an example belongs to. This is done by converting a logistic regression model that predict probability into a binary classification model that predicts one of two classes

### Threshold

To make the conversion from logistic regression model to a classification model, a threshold probability - also called classification threshold - is chosen (value between 0 and 1). Examples with a probability above the threshold value are then assigned a positive class (example: spam), whereas examples with probability with a lower value than the threshold are assigned a negative class, teh alternative class (example: not spam).
Handling the case where the predicted score is equal to the threshold depends on the implementation chosen for the classification model.

### Confusion Matrix

The probability score is not reality or ground truth. If you lay out the ground truth as columns and the model's predictions as rows, the resulting table is called confusion matrix. There are four possible outcomes for each output from a binary classifier.

1. True Positive (TP): a spam email correctly classified as a spam email (spam emails automatically sent to spam folder)
2. False Positive (FP): a not-spam email misclassified as spam (legit emails sent to spam folder)
3. True Negative (TN): a spam email misclassified as not-spam (spam emails that go to inbox; not caught by spam filter)
4. False Negative (FN): a not-spam email classified as not-spam (legit emails sent to inbox)

The total in each row gives all predicted positives (TP + FP) and all predicted negatives (TN + FN) regardless of validity. The total in each column gives all real positives (TP + FN) and all real negatives (FP + TN) regardless of model classification. When the total of real positives is not equal to the number of real negatives, the dataset is considered imbalanced.

Different thresholds usually result in different numbers of true and false positives, and true and false negatives.

As the threshold increases, the model will likely predict fewer positives overall, both true and false. A spam classifier with a threshold of .9999 will only label an email as spam if it considers the classification to be at least 99.99% likely, which means it is highly unlikely to mislabel a legitimate email, but also likely to miss actual spam email.
As the threshold increases, the model will likely predict more negatives overall, both true and false. At a very high threshold, almost all emails, both spam and not-spam, will be classified as not-spam.

### Metrics

[Jump to Classification Metrics](#classification-metrics)

### ROC and AUC

Model metrics evaluate a model at a single classification threshold value. There are different tools that can evaluate a model's quality across all possible thresholds.

#### Receiver-Operating Characteristic curve (ROC)

The ROC curve is a visual representation of the model's performance across all thresholds. It is drawn by calculating the TPR and the FPR at every possible threshold (in practice, at selected intervals), then graphing TPR over FPR.

The points on a ROC curve closest to the point (0,1) represent the best-performing thresholds for that model.

#### Area Under the Curve (AUC)

This represents the probability that a model, given a randomly chosen positive and negative example, will rank the positive higher than the negative.

Example: a spam classifier with AUC of 1.0 always assigns a random spam email a higher probability of being spam than a random legitimate email

AUC is a good measure for comparing performance of two different models, as long as the dataset is roughly balanced. The model with the greater AUC is generally the better one.

### Prediction Bias

This is a quick check that can flag issues with the model or training data early on. Prediction bias is the difference between the mean of the model's predictions and the mean of the ground-truth labels in the data.
Prediction bias can be caused by:

- biases or noise in the data
- too strong regularization (model unnecessarily oversimplified)
- bugs in the model training pipeline
- set of features were insufficient

### Multi-class Classification

Multi-class classification can be treated as an extension of binary classification to more than two classes.
If class membership isn't exclusive, which is to say, an example can be assigned to multiple classes, this is known as a multi-label classification problem.

</details>

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

True and false positives and negatives are used to calculate several useful metrics for evaluating models. Which evaluation are most meaningful depends on the specific model and the specific task, the cost of different misclassifications, and whether the dataset is balanced or imbalanced. Also, the metrics change when the threshold changes so very often the user tunes the threshold to optimize one of these metrics.

### Classification Metrics

#### Accuracy

Accuracy is the proportion of all classifications that are correct, whether positive or negative; matehmatically defined as:

$$
\text{Accuracy} = \frac{\text{correct classifications}}{\text{total classifications}} = \frac{TP + TN}{TP + FP + TN + FN}
$$

A perfect model would have zero false positives and zero false negatives, and therefore an accuracy of 1.0, or 100%.
Because it incorporates all four outcomes from the confusion matrix (TP, FP, TN, FN), given a balanced dataset, with similar numbers of examples in both classes, accuracy can serve as a coarse-grained measure of model quality. For this reason, it is often the default evaluation metric used for generic or unspecified models carrying out generic or unspecified tasks.
However, when the dataset is imbalanced, or where one kind of mistake (FN or FP) is more costly than the other, which is the case in most real-world applications, it's better to optimize for one of the other metrics instead.
For heavily imbalanced datasets, where one class appears very rarely, say 1% of the time, a model that predicts negative 100% of the time would score 99% on accuracy, despite being useless.

#### Precision

Precision is the proportion of all model's positive classifications that are actually positive, mathematically defined as:

$$
\text{Precision} = \frac{\text{correctly classified actual positives}}{\text{everything classified as positive}} = \frac{TP}{TP + FP}
$$

Example: precision measures the fraction of emails classified as spam that were actually spam.

A hypothetical perfect model would have zero false positives and therefore a precision of 1.0.

In an imbalanced dataset where the number of actual positives is very, very low, say 1-2 examples in total, precision is less meaningful and less useful as a metric.

Precision improves as false positives decrease, while recall improves when false negatives decrease. But increasing the classification threshold tends to decrease the number of false positives and increase the number of false negatives, while decreasing the threshold has the opposite effects. As a result, precision and recall often show an inverse relationship, where improving one of them worsens the other.

#### Recall (or True Positive Rate)

Recall or TPR is the proportion of all actual positives that were classified correctly as positives, mathematically defined as:

$$
\text{Recall} = \frac{\text{correctly classified actual positives}}{\text{all actual positives}} = \frac{TP}{TP + FN}
$$

False negatives are actual positives that were misclassified as negatives, which is why they appear in the denominator.

Example: recall measures the fraction of spam emails that were correctly classified as spam. This is why another name for recall is probability of detection: it answers the question "What fraction of spam emails are detected by this model".

A hypothetical perfect model would have zero false negatives and therefore a recall (TPR) of 1.0, which is to say, a 100% detection rate.

In an imbalanced dataset where the number of actual positives is very low, recall is a more meaningful metric than accuracy because it measures the ability of the model to correctly identify all positive instances.

#### F1-Score

F1-score is the harmonic mean of precision and recall, mathematically defined as:

$$
\text{F1} = 2 * \frac{\text{precision * recall}}{\text{precision + recall}} = \frac{2TP}{2TP + FP + FN}
$$

This metric balances the importance of precision and recall, and is preferable to accuracy for class-imbalanced datasets. When precision and recall both have perfect scores of 1.0, F1 will also have a perfect score of 1.0. More broadly, when precision and recall are close in value, F1 will be close to their value. When precision and recall are far apart, F1 will be similar to whichever metric is worse.

#### False Positive Rate (FPR)

The FPR is the proportion of all actual negatives that were classified incorrectly as positives, also called probability of false alarm, mathematically defined as:

$$
\text{FPR} = \frac{\text{incorrectly classified actual negatives}}{\text{all actual negatives}} = \frac{FP}{FP + TN}
$$

False positives are actual negatives that were misclassified, which is why they appear in the denominator.

Example: FPR measures the fraction of legitimate emails that were incorrectly classified as spam, or the model's rate of false alarms.

A perfect model would have zero false positives and therefore a FPR of 0.0, which is to say, a 0% false alarm rate.

For an imbalanced dataset, FPR is generally a more informative metric than accuracy. However, if the number of actual negatives is very low, FPR may not be an ideal choice, due to its volatility. For example, if there are only four actual negatives in a dataset, one misclassification results in an FPR of 25%, while a second misclassification causes the FPR to jump to 50%. In cases like this, precision may be a more stable metric for evaluating the effects of false positives.

#### Choice of Metrics and Tradeoffs

**Accuracy:**

- Use as a rough indicator of model training progress/convergence for balanced datasets.
- For model performance, use only in combination with other metrics.
- Avoid for imbalanced datasets. Consider using another metric.

**Recall (or True positive rate):**

- Use when false negatives are more expensive than false positives.

**False positive rate:**

- Use when false positives are more expensive than false negatives.

**Precision:**

- Use when it's very important for positive predictions to be accurate.

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
