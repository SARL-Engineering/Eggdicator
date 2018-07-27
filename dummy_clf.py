import glob
import matplotlib as plt
import numpy as np
import os
import PIL
import pickle
import random
import sys

from skimage.feature import hog
from skimage import color, exposure, io
from sklearn import svm, metrics
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split, GridSearchCV


# Constants
GOOD_IMG_DIR = "./good_img_train/"
BAD_IMG_DIR = "./bad_img_train/"
GRID_SEARCH = False

def main():
	print("Running Embryo HOG+SVM Proof of Concept Test...")

	good_img_fns = glob.glob(GOOD_IMG_DIR + "*.jpg")
	bad_img_fns = glob.glob(BAD_IMG_DIR + "*.jpg")
	dataset = []

	print("Reading and calculating HOG of good images...")
	for i, img_fn in enumerate(good_img_fns):

		sys.stdout.write("\r%d%%" % (i*100 / len(good_img_fns)))
		sys.stdout.flush()

		img = io.imread(img_fn, as_grey=True)
		fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True, block_norm="L2-Hys")
		dataset.append([fd, 1])

	print("Reading and calculating HOG of bad images...")
	for i, img_fn in enumerate(bad_img_fns):

		sys.stdout.write("\r%d%%" % (i*100 / len(bad_img_fns)))
		sys.stdout.flush()

		img = io.imread(img_fn, as_grey=True)
		fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True, block_norm="L2-Hys")
        print(fd)
        dataset.append([fd, 0])

	print("Generating feature vectors...")
	random.shuffle(dataset)

	X = []
	y = []

	for i in dataset:
		X.append(i[0])
		y.append(i[1])

	X = np.array(X)
	y = np.array(y)

	print("X dimensions: {}".format(X.shape))
	print("y dimensions : {}".format(y.shape))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
		random_state=0)

	print("Performing Support Vector Machine fitting...")

	tuned_parameters = [{"kernel":["poly"], 
		"C":[0.1, 0.3, 0.6, 0.9]},
		{"kernel":["linear"], "C":[0.1, 0.3, 0.6, 0.9]}]
	scores = ["precision", "recall"]

	if GRID_SEARCH:

		for score in scores:

			print("Tuning hyper-parameters for {}".format(score))
			classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, 
				scoring="{}_macro".format(score))
			classifier.fit(X_train, y_train)

			print("Best parameters found on test set:\n")
			print(classifier.best_params_)
			print("\nGrid scores on test set:\n")

			means = classifier.cv_results_["mean_test_score"]
			stds = classifier.cv_results_["std_test_score"]

			for mean, std, params in zip(means, stds, 
				classifier.cv_results_["params"]):
				print("%0.3f (+/-%0.03f for %r" % (mean, std*2, params))

			print("\nDetailed classification report:\n")
			y_true, y_pred = y_test, classifier.predict(X_test)
			print(metrics.classification_report(y_true, y_pred))

		

	else:

		classifier = svm.SVC(kernel="poly", C=100)
        print(X_train)
        print(y_train)
        classifier.fit(X_train, y_train)
        print("Performing Support Vector Machine performance test...")
        expected = y_test
        predicted = classifier.predict(X_test)
        
        print("Classification report for classifier %s:\n%s\n"
			% (classifier, metrics.classification_report(expected, predicted)))
        pickle.dump(classifier, open("./egg_clf.p", "wb+"))

if __name__ == '__main__':
	main()


