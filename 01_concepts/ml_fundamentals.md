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

Types of Losses in Linear Regression

1. **L₁ Loss**: sum of the absolute values of the difference between the actual values and the predicted values
   L₁ = |actual₁ - predicted₁| + |actual₂ - predicted₂| + ... + |actualₙ - predictedₙ|

2. **Mean Absolute Error (MAE)**: the average of L₁ losses over a set of N examples
   MAE = (1/N) × Σ |actual - predicted|

3. **L₂ Loss**: sum of squared difference between predicted and actual values
   L₂ = (actual₁ - predicted₁)² + (actual₂ - predicted₂)² + ... + (actualₙ - predictedₙ)²

4. **Mean Squared Error (MSE)**: the average of L₂ losses across a set of N examples
   MSE = (1/N) × Σ (actual - predicted)²

5. **Root Mean Squared Error (RMSE)**: the square root of the mean squared error (MSE)
   RMSE = √(MSE) = √[(1/N) × Σ (actual - predicted)²]

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
Log Loss = -(1/N) × Σ [y × log(ŷ) + (1-y) × log(1-ŷ)]

where,

- $N$ is the number of labeled examples in the dataset
- y is the label for the ith example (must be 0 or 1)
- ŷ is your model's prediction for the ith example (between 0 and 1), given the set of features in x

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

Standard supervised learning pipeline:

1. **Load & Inspect Data** - Import dataset, check shape, types, missing values, target distribution
2. **Clean Data** - Handle missing values, remove duplicates, fix outliers, validate data quality
3. **Feature Engineering** - Encode categorical variables, scale numerical features, create new features
4. **Split Data** - Train/validation/test split (or just train/test for simple projects)
5. **Train Model** - Fit model on training data
6. **Evaluate** - Check performance on validation/test set using appropriate metrics
7. **Check Overfitting** - Compare train vs test scores (gap >5% suggests overfitting)
8. **Iterate** - Adjust features, hyperparameters, or try different models based on results
9. **Final Test** - Confirm performance on held-out test set
10. **Deploy** - Use model on real-world data

In summary, load, clean, engineer, split, train, evaluate, iterate until test performance is good.

---

## Train/Test Split

**Quick summary**: Split data before any processing. Typical: 60% train / 20% validation / 20% test. Fit transformations on train only, then apply to validation/test. See "Dividing the Original Dataset" section for full details.

**Data leakage**: Using test/validation data to influence training (e.g., fitting scaler on entire dataset before splitting). This gives falsely optimistic results.

---

## Feature Engineering

Process of transforming raw data into features that help models learn better.

### Handling Missing Values

**Options**:

- **Delete**: If <5% missing and enough data remains
- **Impute numerical**: Use mean/median (median better for outliers)
- **Impute categorical**: Use mode or create "missing" category
- **Add indicator**: Boolean column showing which values were imputed

**Best practice**: Document which values are imputed (model should know the difference)

_See "Numerical Data: Scrubbing" and "Complete vs Incomplete Examples" sections for details._

### Encoding Categorical Variables

**Why needed**: Models only understand numbers, not strings.

**Methods**:

- **Label Encoding**: Assign integers (0, 1, 2...) - use for ordinal data (small/medium/large)
- **One-Hot Encoding**: Binary vector for each category - use for nominal data (red/blue/green)
- **Target Encoding**: Replace category with mean of target - use carefully (risk of leakage)

**One-hot example**:

- Color: Red -> [1, 0, 0]
- Color: Blue -> [0, 1, 0]
- Color: Green -> [0, 0, 1]

_See "Categorical Data" section for vocabulary, sparse representation, and OOV handling._

### Feature Scaling

**Why needed**: Features on different scales (age: 0-100, income: 0-100000) cause training issues.

**Methods**:

- **Min-Max Scaling (Linear)**: Scale to [0, 1] range - use for uniform distributions
- **Standardization (Z-score)**: Mean=0, StdDev=1 - use for normal distributions (most common)
- **Log Scaling**: log(x) - use for power-law distributions (highly skewed data)

**When to use**:

- **Always for**: Linear models, logistic regression, SVM, neural networks
- **Not needed for**: Tree-based models (Random Forest, XGBoost)

**CRITICAL**: Fit scaler on TRAIN set only, then transform train/test/validation.

_See "Numerical Data: Normalization" section for formulas and detailed guidance._

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

Average absolute difference: |predicted - actual|. If MAE = $20k, predictions are off by $20k on average. Easy to interpret (same units as target). More info in "Linear Regression" section.

#### RMSE (Root Mean Squared Error)

Square root of mean squared error. Penalizes large errors more heavily than MAE due to squaring. More sensitive to outliers. Also more info in "Linear Regression" section.

#### R² (R-squared)

Proportion of variance explained by model (0 to 1). R² = 0.85 means model explains 85% of price variation. Remaining 15% is unexplained variance.

---

## Data

The model ingests an array of floating-point values called a feature vector. The feature vector doesn't use the dataset's raw values, but instead processed feature values that represent these raw values.

Feature engineering is determining the best way to represent raw dataset values as trainable values in the feature vector. The most common feature engineering techniques are:

- Normalization: converting numerical values into a standard range
- Binning/Bucketing: converting numerical values into buckets of ranges

Detecting Outliers in a Dataset:

- the standard deviation is almost as high as the mean
- the delta between 75% and max is much higher than the delta between min and 25%.

### Numerical Data: Normalization

The goal of normalization is to transform features to be on similar scale.

Benefits of normalization:

- helps model converge more quikcly during training
- helps model infer more useful predictions
- helps model avoid NaN trap when feature values are very high (when a value exceeds floating-point precision limit, it is set to NaN instead of a number)
- helps model learn appropriate weights for each feature

Three popular normalization methods:

- Linear Scaling
- Z-score scaling
- Log scaling

#### Linear Scaling

Linear scaling (more commonly scaling) means converting floating-point values from their natural range into a standard range (usually 0 to 1 or -1 to +1). Its formula: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$, where:

- $x'$ is the scaled value
- $x$ is the original value
- $x_{\min}$ is the lowest value in the dataset of this feature
- $x_{\max}$ is the highest value in the dataset of this feature

Linear scaling is a good choice when all of these conditions are met:

1. The lower and upper bounds of your data don't change much over time (human age typically between 0 and 100)
2. The feature contains few or no outliers, and those outliers aren't extreme (only 3% of population over 100)
3. The feature is approximately uniformly distributed across its range. That is, a histogram would show roughly even bars for most values.

Most real-world features do not meet all of the criteria for linear scaling. Z-score scaling is typically a better normalization choice than linear scaling. Linear scaling better for uniform distributions (flat-shaped), while z-score scaling is better for normal distributions (bell-shaped, peak close to mean).

#### Z-score Scaling

A Z-score is the number of standard deviations a value is from the mean. Representing a feature with Z-score scaling means storing that feature's Z-score in the feature vector. Its formula: $x' = \frac{x - \mu}{\sigma}$, where

- $x'$ is the z-score
- $x$ is the raw value
- $\mu$ is the mean
- $\sigma$ is the standard deviation

#### Log Scaling

Log scaling computes the logarithm of the raw value. In practice, log scaling usually calculates the natural logarithm (ln). Its formula: $x' = \ln(x)$.

Log scaling is helpful when the data conforms to a power law distribution. A power law distribution looks as follows:

- Low values of X have very high values of Y
- As the values of X increase, the values of Y quickly decrease. Consequently, high values of X have very low values of Y.

Log scaling can be used when the feature distribution is heavy skewed on at least either side of tail (heavy tail-shaped).

#### Clipping (not a true normalization technique, but can be very effective)

Clipping is a technique to minimize the influence of extreme outliers. It usually reduces the value of outliers to a specific maximum value.
Example: clipping the feature value at 4.0 doesn't mean that your model ignores all values greater than 4.0. Rather, it means that all values that were greater than 4.0 now become 4.0.
Clipping can be used when training a model and/or after using normalization (for example, if few outliers have absolute values far greater than 3).

### Numerical Data: Binning

Binning/bucketing is a feature engineering technique that groups different numerical subranges into bins or buckets. In many cases, binning turns numerical data into categorical data. A model trained on these bins will react no differently to feature values if they are in the same bin. Even though a feature is a single column in the dataset, binning causes a model to treat that feature as $n$ separate features (where $n$ is the number of bins). Therefore, the model learns separate weights for each bin. A model can only learn the association between a bin and a label if there are enough examples in that bin.

Binning is a good alternative to scaling or clipping when either of the following conditions is met:

1. The overall linear relationship between the feature and the label is weak or nonexistent.
2. When the feature values are clustered.

Binning transforms numerical data into categorical data.

#### Quantile Bucketing

Quantile bucketing creates bucketing boundaries such that the number of examples in each bucket is exactly or nearly equal. Quantile bucketing mostly hides the outliers.

Bucketing with equal intervals works for many data distributions. For skewed data, however, quantile bucketing is more useful. Equal intervals give extra information space to the long tail while compacting the large torso into a single bucket. Quantile buckets give extra information space to the large torso while compacting the long tail into a single bucket.

### Numerical Data: Scrubbing

Many examples in datasets are unreliable due to one or more of the following problems:

- Omitted values
- Duplicate examples
- Out-of-range feature values
- Bad labels

### Qualities of Good Numerical Features

1. Clearly named
2. Checked or tested before training
3. Sensible

### Numerical Data: Polynomial Transforms

Polynomial transforming is a trick where you don’t change the model, you just change what you feed into it. A linear model is “linear” not in terms of the original real-world relationship, but in terms of the features it sees. When you square a feature (or cube it, etc.), you’re creating a new column of data that the model treats like any other input. From the model’s point of view, it’s still doing a weighted sum of features (add this times a weight, plus that times a weight) which is why training with gradient descent works exactly the same.
What changes is the shape the model can represent in the original space. A straight line in the transformed feature space becomes a curve when you map it back to the original variable. So even though the model is mathematically linear, it can draw curved boundaries in the real world because the features themselves are nonlinear transformations of the data.

Example: if your features are: original value: $x$ and synthetic feature: $x²$
The model sees:

- feature 1: some number
- feature 2: another number

Synthetic features can be created to model non-linear relationships between two features. These synthetic features can then be used as inputs to a linear model to enable it to represent nonlinearities.

### Categorical Data

Categorical data has a specific set of possible values (could be strings like in color names, or numbers like in bins or postal codes).
**Encoding means**converting categorical or other data to numerical vectors that a model can train on. This conversion is necessary because models can only train on floating-point values.

#### Vocabulary and One-hot Encoding

The term **dimension** is a synonym for the number of elements in a feature vector. Some categorical features are low dimensional.
When a categorical feature has a low number of possible categories, you can encode it as a **vocabulary**. With a vocabulary encoding, the model treats each possible categorical value as a separate feature. During training, the model learns different weights for each category.

**Index numbers**:

Machine learning models can only manipulate floating-point numbers. Therefore, you must convert each string to a unique index number, starting from zero. After converting strings to unique index numbers, you'll need to process the data further to represent it in ways that help the model learn meaningful relationships between the values. If the categorical feature data is left as indexed integers and loaded into a model, the model would treat the indexed values as continuous floating-point numbers.

**One-hot Encoding**:

The next step in building a vocabulary is to convert each index number to its one-hot encoding. In a one-hot encoding each category is represented by a vector (array) of N elements, where N is the number of categories. Exactly one of the elements in a one-hot vector has the value 1.0; all the remaining elements have the value 0.0. It is the one-hot vector, not the string or the index number, that gets passed to the feature vector. The model learns a separate weight for each element of the feature vector.

In a true one-hot encoding, only one element has the value 1.0. In a variant known as **multi-hot encoding**, multiple values can be 1.0.

**Sparse Representation**

A feature whose values are predominantly zero (or empty) is termed a sparse feature. Many categorical features tend to be sparse features. Sparse representation means storing the position of the 1.0 in a sparse vector.
Notice that the sparse representation consumes far less memory than the N-element one-hot vector. Importantly, the model must train on the one-hot vector, not the sparse representation.

The sparse representation of a multi-hot encoding stores the positions of all the nonzero elements.

**Encoding high-dimensional categorical features**
Some categorical features have a high number of dimensions. When the number of categories is high, one-hot encoding is usually a bad choice. Embeddings are usually a much better choice. Embeddings substantially reduce the number of dimensions, which benefits models in two important ways:

1. The model typically trains faster.
2. The built model typically infers predictions more quickly. That is, the model has lower latency.
   Hashing (also called the hashing trick) is a less common way to reduce the number of dimensions.

#### Outliers in Categorical Data

Like numerical data, categorical data also contains outliers. Rather than giving each of these outlier examples a separate category, you can lump them into a single "catch-all" category called out-of-vocabulary (OOV). In other words, all the outlier examples are binned into a single outlier bucket. The system learns a single weight for that outlier bucket.

#### Common Issues with Categorical Data

Data manually labeled by human beings is often referred to as gold labels, and is considered more desirable than machine-labeled data for training models, due to relatively better data quality. This doesn't necessarily mean that any set of human-labeled data is of high quality. Human errors, bias, and malice can be introduced at the point of data collection or during data cleaning and processing (should be checked before training). Any two human beings may label the same example differently. The difference between human raters' decisions is called inter-rater agreement. You can get a sense of the variance in raters' opinions by using multiple raters per example and measuring inter-rater agreement.

Machine-labeled data, where categories are automatically determined by one or more classification models, is often referred to as silver labels. Machine-labeled data can vary widely in quality. Data should be checked not only for accuracy and biases but also for violations of common sense, reality, and intention.

#### Feature Crossing

Feature crosses are created by taking the Cartesian product of two or more categorical or bucketed features of the dataset. Like polynomial transforms, feature crosses allow linear models to handle nonlinearities. Feature crosses also encode interactions between features.

Feature crosses are somewhat analogous to Polynomial transforms. Both combine multiple features into a new synthetic feature that the model can train on to learn nonlinearities. Polynomial transforms typically combine numerical data, while feature crosses combine categorical data.

Domain knowledge can suggest a useful combination of features to cross. Without that domain knowledge, it can be difficult to determine effective feature crosses or polynomial transforms by hand. It's often possible, if computationally expensive, to use neural networks to automatically find and apply useful feature combinations during training.

Crossing two sparse features produces an even sparser new feature than the two original features. For example, if feature A is a 100-element sparse feature and feature B is a 200-element sparse feature, a feature cross of A and B yields a 20,000-element sparse feature.

## Datasets, Generalization, and Overfitting

### Types of Data

A dataset could contain many kinds of datatypes including:

- numerical data
- categorical data
- human language, including words, sentences, entire text documents
- multimedia (such as images, videos, and audio files)
- outputs from other ML systems
- embedding vectors

### Quantity of Data

As a rule of thumb (not strict law) model should train on at least an order of magnitude (10x) or two (100x) more examples than trainable parameters. However, good models generally train on substantially more examples than that.
Models trained on large datasets with few features generally outperform models trained on small datasets with a lot of features.

It's possible to get good results from a small dataset if you are adapting an existing model already trained on large quantities of data from the same schema.

### Quality and Reliability

A **high-quality** dataset helps your model accomplish its goal. A low quality dataset inhibits your model from accomplishing its goal.
A high-quality dataset is usually also reliable. **Reliability** refers to the degree to which you can trust your data. A model trained on a reliable dataset is more likely to yield useful predictions than a model trained on unreliable data.

Common causes of unreliable data:

- Omitted values
- Duplicate examples
- Bad feature values
- Bad labels
- Bad sections of data

Automation can be used to flag unreliable data (e.g., unit tests for out of range values).

#### Complete vs Incomplete Examples

In a perfect scenario, each example is complete: each feature of an example has a value. In real world, most examples are incomplete: at least one feature value is missing. Training should be on complete examples. To fix or eliminate incomplete examples do one of the following:

1. Delete incomplete examples
2. Impute missing values (providing well-reasoned guesses for the missing values)

If the dataset contains enough complete examples to train a useful model, then consider deleting the incomplete examples. If only one feature is missing a significant amount of data and that one feature probably can't help the model much, then consider deleting that feature if the model works just or almost as well without it. Conversely, if you don't have enough complete examples to train a useful model, then you might consider imputing missing values.

If you can't decide whether to delete or impute, consider building two datasets: one formed by deleting incomplete examples and the other by imputing. Then, determine which dataset trains the better model.

One common algorithm for imputation is to use the mean or median as the imputed value. Consequently, when you represent a numerical feature with Z-scores, then the imputed value is typically 0 (because 0 is generally the mean Z-score).

Imputed values are rarely as good as the actual values. Therefore, a good dataset tells the model which values are imputed and which are actual. One way to do this is to add an extra Boolean column to the dataset that indicates whether a particular feature's value is imputed.Then, during training, the model will probably gradually learn to trust examples containing imputed values for feature temperature less than examples containing actual values.

#### Direct vs Proxy Labels

Direct labels are labels that are exactly what the model is trying to predict and already exist as a column in the dataset. Direct labels are preferred and should be used whenever they are available.

Proxy labels are labels that are related to, but not identical to, the prediction target. Proxy labels are imperfect approximations, and models trained on them are only as good as the strength of the relationship between the proxy and the true prediction.

Sometimes a direct label does not exist or cannot be easily represented as a numeric floating-point value, which machine learning models require. In these cases, a proxy label is used as a practical compromise.

### Class-balanced datasets vs class-imbalanced datasets

Consider a dataset containing a categorical label whose value is either the positive class or the negative class. In a class-balanced dataset, the number of positive classes and negative classes is about equal.

In a class-imbalanced dataset, one label is considerably more common than the other. In the real world, class-imbalanced datasets are far more common than class-balanced datasets.In a class-imbalanced dataset the more common label is called the majority class, and the less common label is called the minority class.

**Difficulty training class-imbalanced datasets**
Training aims to create a model that successfully distinguishes the positive class from the negative class. To do so, batches need a sufficient number of both positive classes and negative classes. That's not a problem when training on a mildly class-imbalanced dataset since even small batches typically contain sufficient examples of both the positive class and the negative class. However, a severely class-imbalanced dataset might not contain enough minority class examples for proper training.

Accuracy is usually a poor metric for assessing a model trained on a class-imbalanced dataset.

**Training a class-imbalanced dataset**
During training, a model should learn two things:

- What each class looks like; what feature values correspond to what class.
- How common each class is; what is the relative distribution of the classes.
  Standard training combines these two goals. The following two-step technique called **downsampling** and **upweighting** the majority class separates these two goals, enabling the model to achieve both goals.

#### First: Downsample the majority class

Downsampling means training on a disproportionately low percentage of majority class examples. That is, you artificially force a class-imbalanced dataset to become somewhat more balanced by omitting many of the majority class examples from training. Downsampling greatly increases the probability that each batch contains enough examples of the minority class to train the model properly and efficiently.

#### Then: Upweight the downsampled class

Downsampling introduces a prediction bias by showing the model an artificial world where the classes are more balanced than in the real world. To correct this bias, you must upweight the majority classes by the factor to which you downsampled. Upweighting means treating the loss on a majority class example more harshly than the loss on a minority class example.

To rebalance your dataset you should experiment with different downsampling and upweighting factors just as you would experiment with other hyperparameters.

Downsampling and upweighting the majority class brings the following **benefits**:

- Better model: The resultant model "knows" both of the following: the connection between features and labels, and the true distribution of the classes
- Faster convergence: During training, the model sees the minority class more often, which helps the model converge faster.

### Dividing the Original Dataset

#### Training, Validation, and Test Sets

You should test a model against a different set of examples than those used to train the model. Testing on different examples is stronger proof of your model's fitness than testing on the same set of examples. You get those different examples by splitting the original dataset. The original dataset is split into two subsets:

- A training set that the model trains on.
- A test set for evaluation of the trained model.

The more often you use the same test set, the more likely the model closely fits the test set, which might make it harder for the model to fit real-world data.

A better approach is to divide the dataset into three subsets. In addition to the training set and the test set, the third subset is:

- A validation set performs the initial testing on the model as it is being trained.

Use the validation set to evaluate results from the training set. After repeated use of the validation set suggests that your model is making good predictions, use the test set to double-check your model.

"Tweaking a model" means adjusting anything about the model; from changing the learning rate, to adding or removing features, to designing a completely new model from scratch.
An ML workflow consists of the following stages:

1. Training model on the training set.
2. Evaluating model on the validation set.
3. Tweaking model according to results on the validation set.
4. Iterate on 1, 2, and 3, ultimately picking the model that does best on the validation set.
5. Confirm the results on the test set.

When you transform a feature in your training set, you must make the same transformation in the validation set, test set, and real-world dataset.
The workflow above is optimal, but even with that, test sets and validation sets still wear out with repeated use. That is, the more you use the same data to make decisions about hyperparameter settings or other model improvements, the less confidence that the model will make good predictions on new data. For this reason, it's a good idea to collect more data to refresh the test set and validation set.

Training and testing are nondeterministic. Sometimes, by chance, your test loss is incredibly low. Rerun the test to confirm the result.Many of the examples in the test set can be duplicates of examples in the training set. This can be a problem in a dataset with a lot of redundant examples. It is strongly recommended to delete duplicate examples from the test set before testing.

In summary, a good test set or validation set meets all of the following criteria:

- Large enough to yield statistically significant testing results.
- Representative of the dataset as a whole (don't pick a test set with different characteristics than the training set)
- Representative of the real-world data that the model will encounter as part of its business purpose.
- Zero examples duplicated in the training set.

Every example used in testing the model is one less example used in training the model. Dividing examples into train/test/validation sets is a zero-sum game. This is the central trade-off.

## Overfitting vs Underfitting

### Generalization

When you train a model on data, you're teaching it to perform well on that specific set of data. But this training data is just a small subset of the real-world data. If a model makes high quality predictions on the training data, but much lower quality predictions on new data, then the model didn't generalize. In this case, it is said that the model has overfit the training data. It has been tuned so closely to specific patterns os the data it's already seen that it can identify patterns in new data. A model must make good predictions on new data. That is, you're aiming to create a model that "fits" new data.
Generalization is the opposite of overfitting. That is, a model that generalizes well makes good predictions on new data. The goal is to create a model that generalizes well to new data.

#### Generalization Conditions

While developing a model, your test set serves as a proxy for real-world data. Training a model that generalizes well implies the following dataset conditions:

- Examples must be independently and identically distributed (examples can't influence each other).
- The dataset is stationary, meaning the dataset doesn't change significantly over time.
- The dataset partitions have the same distribution (examples in the training set are statistically similar to the examples in the validation set, test set, and real-world data)

### Overfitting

Overfitting means creating a model that matches (memorizes) the training set so closely that the model fails to make correct predictions on new data. An overfit model makes excellent predictions on the training set but poor predictions on new data.

#### Detecting Overfitting

The following curves help you detect overfitting:

- loss curves
- generalization curves
  A loss curve plots a model's loss against the number of training iterations. A graph that shows two or more loss curves is called a generalization curve.
  If the two loss curves (on training data and validation data) behave similarly at first and then diverge; meaning after a certain number of iterations, loss declines or holds steady (converges) for the training set, but increases for the validation set, this suggests overfitting.
  In contrast, a generalization curve for a well-fit model shows two loss curves that have similar shapes.

#### Overfitting Causes

Very broadly speaking, overfitting is caused by one or both of the following problems:

1. The training set doesn't adequately represent real life data (or the validation set or test set).

- The model is too complex.

#### Model Complexity

Complex models typically outperform simple models on the training set. However, simple models typically outperform complex models on the test set (which is more important).

#### Regularization

Machine learning models must simultaneously meet two conflicting goals:

- Fit data well.
- Fit data as simply as possible.
  One approach to keeping a model simple is to penalize complex models; to force the model to become simpler during training. Penalizing complex models is one form of regularization.

#### Loss and complexity

So far, it's suggested that the only goal when training was to minimize loss; that is: $minimize(loss)$
Models focused solely on minimizing loss tend to overfit. A better training optimization algorithm minimizes some combination of loss and complexity: $minimize(loss+complexity)$
Unfortunately, loss and complexity are typically inversely related. As complexity increases, loss decreases. As complexity decreases, loss increases. A reasonable middle ground should be found where the model makes good predictions on both the training data and real-world data. That is the model should find a reasonable compromise between loss and complexity.

**What is complexity?**

Complexity is a function of the model's weights. This is one way to measure some models' complexity. This metric is called **L1 regularization**.
Complexity is also a function of the square of the model's weights. This metric is called **L2 regularization**.

**L2 Regularization**

$L_2$ regularization is a popular regularization metric, which uses the following formula:

$$
L_2 \text{ regularization} = w_1^2 + w_2^2 + \cdots + w_n^2
$$

Weights close to zero don't affect $L_2$ regularization much, but large weights can have a huge impact. $L_2$ regularization encourages weights toward 0, but never pushes weights all the way to zero. Since $L_2$ regularization encourages weights towards 0, the overall complexity will probably drop.

**Regularization Rate (Lambda)**

Training attempts to minimize some combination of loss and complexity: $minimize(loss+complexity)$.
Model developers tune the overall impact of complexity on model training by multiplying its value by a scalar called the regularization rate. The Greek character lambda typically symbolizes the regularization rate.
That is, model developers aim to do the following: $\text{minimize}(\text{loss} + \lambda\, \text{complexity})$

A high regularization rate:

- Strengthens the influence of regularization, thereby reducing the chances of overfitting.
- Tends to produce a histogram of model weights having: a normal distribution, and a mean weight of 0.
  A low regularization rate:
- Lowers the influence of regularization, thereby increasing the chances of overfitting.
- Tends to produce a histogram of model weights with a flat distribution.

Setting the regularization rate to zero removes regularization completely. In this case, training focuses exclusively on minimizing loss, which poses the highest possible overfitting risk.

**Picking the regularization rate**

The ideal regularization rate produces a model that generalizes well to new, previously unseen data. That ideal value is data-dependent, so you must do some tuning.
**Early stopping:** an alternative to complexity-based regularization
Early stopping is a regularization method that doesn't involve a calculation of complexity. Instead, early stopping simply means ending training before the model fully converges. For example, you end training when the loss curve for the validation set starts to increase (slope becomes positive). Although early stopping usually increases training loss, it can decrease test loss. Early stopping is a quick, but rarely optimal, form of regularization. The resulting model is very unlikely to be as good as a model trained thoroughly on the ideal regularization rate.

**Finding equilibrium between learning rate and regularization rate**
Learning rate and regularization rate tend to pull weights in opposite directions. A high learning rate often pulls weights away from zero, and a high regularization rate pulls weights towards zero.
If the regularization rate is high with respect to the learning rate, the weak weights tend to produce a model that makes poor predictions. Conversely, if the learning rate is high with respect to the regularization rate, the strong weights tend to produce an overfit model.

Examples:

For an oscillating loss curve, you can try:

- Reducing the learning rate.
- Reducing the training set to a tiny number of trustworthy examples.
- Checking your data against a data schema to detect bad examples, and then remove the bad examples from the training set.

For a loss curve with a sharp jump, causes can include:

- The input data contains a burst of outliers.
- The input data contains one or more NaNs—for example, a value caused by a division by zero.

For curves where test loss diverges from training loss, reason can be:

- The model is overfitting the training set.
  Possible solutions:
- Make the model simpler, possibly by reducing the number of features.
- Increase the regularization rate.
- Ensure that the training set and test set are statistically equivalent.

For a loss curve that gets stuck, a possible cause can be:

- The training set is not shuffled well. For example, a training set that contains 100 images of dogs followed by 100 images of cats may cause loss to oscillate as the model trains. Ensure that you shuffle examples sufficiently.

### Underfitting

An underfit model doesn't even make good predictions on the training data. Underfitting is producing a model with poor predictive ability because the model hasn't fully captured the complexity of the training data. Many problems can cause underfitting, including:

- Training on the wrong set of features.
- Training for too few epochs or at too low a learning rate.
- Training with too high a regularization rate.
- Providing too few hidden layers in a deep neural network.

---

## Model Selection

### When to Use What

**Logistic Regression** (Classification):

- Baseline model, fast, interpretable
- Use for: linear decision boundaries, need to explain predictions
- Avoid for: highly non-linear data

**Random Forest** (Both):

- Handles non-linearity, feature interactions
- Use for: complex patterns, don't need interpretability
- Avoid for: need to explain to stakeholders, very small datasets

**Linear Regression** (Regression):

- Baseline, interpretable, fast
- Use for: linear relationships, need coefficients
- Avoid for: clear non-linear patterns

**Strategy**: Start simple (Linear/Logistic), check performance. If insufficient, try Random Forest. Compare train vs test scores to check overfitting.

---

## Key Takeaways
