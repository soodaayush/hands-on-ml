from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# This imports a Stochastic Gradient Descent classifier

# Gradient descent: a model's error is like a landscape, you find the lowest
# point (least error) by repeatedly checking the slope and stepping downhill.
# Basically a model is racing to the bottomest part of the land it can find
# to achieve the least error
# Stochastic Gradient Descent checks one random example per step to determine
# if its on the right tracker rather than measuring slope. This is much
# faster, but results in a zaggy path downhill
# Ideal for classification as there are many images to process

from sklearn.linear_model import SGDClassifier

# Cross validation splits our training data into several chunks and trains
# the model on some chunks and tests on the rest. It repeats this process
# several times and averages the scores of how accurate your model is at
# predicting based off of the data

# cross_val_score runs cross-validation for us and gives back the accuracy
from sklearn.model_selection import cross_val_score

# This is a fake "do-nothing-smart" model scikit learn offers as a baseline
# model to compare to. It ignores the feature inputs entirely and makes
# predictions using a dumb strategy. We can determine how accurate the SGD
# classfier is compared to the dummy classifier

from sklearn.dummy import DummyClassifier

# Cross validation splits your training data into several chunks (folds). It
# trains the model on some folds and uses the rest for testing, then repeats
# this process while excluding a different chunk each iteration. It then
# averages out the results of predictions in terms of their accuracies

# cross_val_predict tell us what the model predicted for each instance by
# providing a full array of predictions

from sklearn.model_selection import cross_val_predict

# A confusion matrix is a table showing exactly what your model got right and
# what it mixed up; not just how many it got right overall but which things
# it confused with which. For example, it shows how many 5s were correctly
# classified as 5s and how many were incorrectly classified
from sklearn.metrics import confusion_matrix

# The two tools below measure how good our model's predictions are
# precision_score measures the percentage of accurate predictions made
# recall_score - of everything that was correct, how much did the model
# predict correctly

# There is a trade-off to this. If you increase precision (meaning that you
# increase the amount of accurate predictions you make), that will reduce the
# recall (the amount of correct answers that were predicted) and vice versa

from sklearn.metrics import precision_score, recall_score

# There is a more nice way of combining precision and recall scores into a
# single metric called the f1 score. higher f1 score = higher precision AND
# recall scores
from sklearn.metrics import f1_score

# This provides a graph that shows us precision and recall scores at each
# threshold. This is useful as it allows us to identify a threshold that
# meets our requirements for what we want precision and recall scores to look
# like
from sklearn.metrics import precision_recall_curve

# The Receiver Operating Characteristic (ROC) curve shows how well your model
# tells positives from negatives by plotting how many real digits (e.g. 5) it
# catches (true positive rate) against how many false alarms it makes (false
# positive rate) as you change the threshold.
from sklearn.metrics import roc_curve

# This imports AUC, or the area under the curve. A perfect classifier will
# have a ROC AUC equal to 1, whereas a purely random classifier will have ROC
# AUC equal to 0.5
from sklearn.metrics import roc_auc_score

# Random Forest builds lots of decision trees. Decision Trees are an
# algorithm a model uses by asking a series of yes/no questions about our
# features, one after the other like a flowchart until it lands on the final
# answer on the bottom (e.g. is grid position < 3, is constructor Red
# Bull/Ferrari/Mercedes -> predict podium or not)

# A Random Forest builds many different decision trees, each trained on
# a slightly different random slice of the data, and has them all vote on the
# final prediction. Majority wins for classification.
from sklearn.ensemble import RandomForestClassifier

# A Support Vector Classifier (SVC) tries to draw the widest gap between
# "num" and "not num" digit images, using only the closest, most borderline
# examples to decide where that boundary goes. Example would be SVC using a 5
# that looks like a 3 and vice versa, to decide where to draw the "5 vs.
# not-5" boundary, ignoring clean, obvious examples entirely

# Number of predictions = (N + (N-1))/2
# N = 10

from sklearn.svm import SVC

# A different way of classifying numbers
# Instead of training a classifier for every pair of classes (e.g. 1 vs.
# non-1), One Vs. Rest (OvR) trains just one classifier per class, where each
# one answers the question: "is this a 5, or is it anything else?"

from sklearn.multiclass import OneVsRestClassifier

# StandardScaler rescales your features so they're centered around 0 with a
# similar spread. This helps many models (like SGD) train better,
# since features on wildly different scales can otherwise confuse the
# learning process

from sklearn.preprocessing import StandardScaler

# Since we are now dealing with 10 classes rather than 2 (remember binary
# classifier), confusion matrices will look a lot more complicated. An easier
# way of looking at it will be to use the confusion matrix display, a more
# visual way of looking at a confusion matrix

from sklearn.metrics import ConfusionMatrixDisplay

# Function for rendering images of digits
def plot_digit(image_data):
    # We resize the image to be 28x28 pixels
    image = image_data.reshape(28, 28)
    # Rendering binary data as a 2D image
    plt.imshow(image, cmap="binary")
    # We disable the axis
    plt.axis("off")

# We are importing the MNIST dataseet, a set of 70,000 small images of digits
# handwritten by high school students and employees of the US Census Bureau
mnist = fetch_openml("mnist_784", as_frame=False)

# Features and labels are being defined
X, y = mnist.data, mnist.target
print(X)
# Prints 70,000 images, each containing 784 features
print(X.shape)
# Contains labels (e.g. 1, 2, 3, 4, 5)
print(y)
# Prints number of labels (70,000)
print(y.shape)

some_digit = X[0] # Binary representation of 5
print(y[0]) # Prints 5
plot_digit(some_digit)
plt.show() # Renders the image

# Since we are machine learning engineers, we must split our data into a
# training and testing set. Fortunately, it is already organized for us,
# as the training set is the first 60,000 images and the test set is the last
# 10,000 images
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# We are now going to create something called a binary classifier. This
# allows us to solely identify one digit. In this case, the classifier
# distinguishes between two categories (or classes): 5, and non-5

# These are arrays containing true and false values to identify whether or
# not a number is 5
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# We initialze our SGD classifier; remember that random_state ensures that a
# specific random order is the same every time you run this code
# We train the SGD using our training data and the binary classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Since we inputted some_digit, which is 5, this outputs true
print(sgd_clf.predict([some_digit]))

# We are running a cross-validation on our SGD Classifier, inputting our
# training features and labels, splitting the data into three folds (cv) and
# measuring for accuracy

# We consistently have at least 95% accuracy, which is good!
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# We are running our dummy classifier
dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))

# Is right about 90% of the time! This is really because about 10% of the
# images are 5s, so if you always guess that an image is not a 5, you will be
# right about 90% of the time
print(cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# This outputs predictions made on each fold of data
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Prints the matrix. Each row represents a class (e.g. 5 and not-5) while
# each column represents a predicted class. First row are non-5 images. 53,
# 982 were correctly classified as non-5s (true negatives) while 687 were
# wrongly classified as 5s (false positives)
# Second row is for 5s. 1891 were wrongly classified as non-5s while 3530
# images were correctly classified as 5s
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

# Because we fed the model the labels for the data (disguised as
# predictions), this causes the confusion_matrix to compare the predictions
# against the actual answers (both y_train_5 and y_train_perfect_predictions
# are the same), you would get a perfect confusion matrix
y_train_perfect_predictions = y_train_5 # Pretend we reached perfection
print(confusion_matrix(y_train_5, y_train_perfect_predictions))

# Prints precision and recall scores of our data. You would think based on
# our perfect confusion matrix that it would output higher values but, it did
# not
print(precision_score(y_train_5, y_train_pred)) # 83.7%
print(recall_score(y_train_5, y_train_pred)) # 65.1%

# Prints the f1 score
print(f1_score(y_train_5, y_train_pred)) # 73.3%

# We are now looking under the hood of what happens in a regression function
# Instead of asking a model for a direction yes/no prediction, this asks for
# the raw score it computes internally for some_digit. Higher score = more
# confident the digit is a 5; lower score = lower confidence
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores) # 2164.22 - pretty high

# We set up a cut-off point, or threshold. If the score computed above is
# less than 0, it is not 5; if higher than 0, is 5
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred) # Prints True since 2164 > 0

# If we raise the threshold, we decrease the recall_score as the barrier for
# entry for being classified as 5 is a lot higher
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred) # Prints True since 2164 > 0

# Returns raw confidence scores of predictions
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

# We feed the true labels and raw scores into the precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Plots precision (y-axis) against threshold (x-axis) using a blue dashed
# line (b--)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)

# Plots recall (y-axis) against threshold (x-axis) using a green solid line (
# g-)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

# Draws a vertical black line at our chosen threshold value (3000)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="Threshold")

plt.show()

# Another way of getting a better understanding of the precision/recall
# trade-off is directly plotting the two against each other
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

plt.show()

# Let's imagine we want to identify the exact threshold needed to ensure our
# model has a 90% precision score

# The line below checks every value in the precisions array, producing a
# True/False value depending on whether or not the precision is 90% or higher
# argmax() - finds the index of the first True in that array, then stops
# searching
idx_for_90_precision = (precisions >= 0.90).argmax()

# We find the threshold required by passing in the first True index into the
# thresholds array
threshold_for_90_precision = thresholds[idx_for_90_precision]

print(threshold_for_90_precision) # 3370.019499144185

# Now we can make all our predictions based on 90% precision
y_train_pred_90 = (y_scores >= threshold_for_90_precision)

print(precision_score(y_train_5, y_train_pred_90)) # 0.90
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print(recall_at_90_precision) # 0.48

# Feed in true labels and raw scores and receive false positive rate (FPR),
# true positive rate (TPR) and the thresholds that produce each pair. Each of
# the three are represented by arrays
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# We know our 90% precision threshold. This finds where that same threshold
# resides on the ROC curve's own threshold list
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()

# Using the index from above to pull out the exact TPR (recall) and FPR
# values that correspond to the 90% precision threshold
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

# Plots full ROC curve - FPR on x-axis, TPR on y-axis
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")

# Plots the diagonal reference line, representing a random/useless
# classifier's curve, as it represents a baseline
plt.plot([0, 1], [0, 1], "k:", label="Random classifier's ROC curve")

# Plots a single black dot ("ko") at the exact point on the curve
# corresponding with the 90% precision threshold
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

plt.show()

print(roc_auc_score(y_train_5, y_scores)) # 0.96

forest_clf = RandomForestClassifier(random_state=42)

# Instead of returning a raw decision score like decision_function,
# this returns probabilities of each digit belonging to 5 or non-5
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

# Prints probability results for the first two instances
# The model predicts that the first image is positive with 89% probability,
# and it predicts that the second image is negative with 99% probability
print(y_probas_forest[:2])

# y_probas_forest has two columns: probability of "not 5" (column 0) and
# probability of "5" (column 1). This just grabs column 1, since that is the
# score we want to build a curve from
y_scores_forest = y_probas_forest[:, 1]

# This computes precision/recall scores across all thresholds, similar to
# what we did with SGD
precisions_forest, recalls_forest, thresholds_forest = (
    precision_recall_curve(y_train_5, y_scores_forest))

# Plots both recall and precision curves; recall on x-axis and precision on
# y-axis
# Solid line - Random Forest
# Dashed line - SGD
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random "
                                                                     "Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.legend()

# When we render the graph, it is obvious that Random Forest is immensely
# better than SGD in terms of having higher precision and recall scores
plt.show()

# We trained the SVM using the original target classes from 0 to 9 (y_train)
# instead of the 5 vs. the rest target classes (y_train_5). Since there are
# 10 classes (now more than 2), scikit learn uses OVO to train 45 binary
# classifiers. We can now make predictions on digits!
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])

# Instead of one model choosing directly among all 10 digits, Scikit-Learn
# trains 45 mini-classifiers, each an expert at telling apart just one specific
# pair of digits (like "3 or 7?"). A new digit gets judged by all 45 of them,
# and whichever digit wins the most of these one-on-one matchups becomes the
# final prediction.
print(svm_clf.predict([some_digit])) # Prints 5, since the digit is 5

some_digit_scores = svm_clf.decision_function([some_digit])

# Prints the scores per each instance (10 scores per instance): one per
# class (0-9). Each class gets a score equal to the number of duels won plus or
# minus a small tweak to break ties

# The highest score is 9.3, corresponding to class 5
print(some_digit_scores.round(2))

class_id = some_digit_scores.argmax()
print(class_id) # Prints 5

print(svm_clf.classes_) # Prints all classes (e.g. 0, 1, 2, etc.)

# Prints the class at the fifth index,  which happens to be 5
print(svm_clf.classes_[class_id])

# Training a OvR Classifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

print(ovr_clf.predict([some_digit])) # Prints 5

# estimators_ - gives you access to list of individual classifiers that OvR
# is trained on under the hood
print(len(ovr_clf.estimators_))

# Training a multi-class SGD Classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit])) # Prints 3, which is incorrect

# You can see here that the classifier is not very confident in its
# predictions: almost all scores are very negative, while class 3 has a
# positive score and 5 has a negative score
print(sgd_clf.decision_function([some_digit]).round())

# Despite that, it gets around 85.8% accuracy on all test folds
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# Training StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

# As a result of scaling, accuracy has increased to 89.1%
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3,
                      scoring="accuracy"))

# Visualizing the confusion matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()