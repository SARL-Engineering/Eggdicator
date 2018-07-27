import cv2
import datetime
import glob
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import re
import serial
import subprocess
import sys
import time
import serialHandler

from skimage.feature import hog
from skimage import color, exposure, io, transform
from sklearn import svm, metrics
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split, GridSearchCV


class EggAI(object):
    """Top-level class which will handle Eggidicator CV & ML

    EggAI is a high-level class and the only one that should be referenced
    outside of the egg_cv module.  It has the following functions:

    1.  Open a trained SVM model
    2.  Train a new SVM model if no current one exists
    3.  Open and return a camera feed
    4.  Poll for the existence of an embryo-like object
    5.  When detected, take an image of an embryo-like object and run it
        through the model
    6.  Open a communication line
    7.  Return image and the results of the model calculation via USB comms

    Attributes:
        classifier - svm.SVC or svm.LinearSVC object model used for embryo
            viability determination
    """

    def __init__(self, model_file_location, serial_device_name,
                 train=False, start=False):

        super(EggAI, self).__init__()

        # Constants
        self.DEBUG = False

        # Flags
        self.kill_flag = False
        
        self.classifier = None

        # Set up our ML classifier
        if train is False:
            print("Opening SVM from file...")
            self.classifier = self._open_classifier(model_file_location)

        else:
            self.classifier = None

        # Open camera feeds & init parameters
        self.us_factor = 0.05
        self.min_area = 300
        self.signal_timer = None
        self.signal_sent = None
        self.tput_timer = None
        self.tput_delay = 5
        self.startup_time = time.time()
        self.startup_delay = 10

        # Start talking to the Red Stick
        """self.red_stick = EggTalker(serial_device_name)
        self.red_stick.open_comms()"""

        # Begin polling cameras for motion (continuous)
        # self.poll_cams()

        def __str__(self):
            return "Egg AI"

    def _open_classifier(self, file_loc):
        # Returns classifier from pickle object
        try:

            clf = pickle.load(open(file_loc, "rb"))

            if type(clf) is not svm.SVC and type(clf) is not svm.LinearSVC:
                raise TypeError

            return clf

        except TypeError as e:
            print("The pickle file is not a scikit-learn SVM.")
            raise e

    def train(self, pos_img_dir, neg_img_dir, optimize=False, tt_split=0.8,
              criterion="recall", criterion_output=0, orientations=9, ppc=(8, 8),
              cpb=(3, 3)):
        """Trains a new SVM model

        Args:
            pos_img_dir - file directory containing positive images
            neg_img_dir - file directory containing negative images
            optimize - if True, run time-intensive grid search operation
            tt_split - ratio of training to test data
            criterion - classification accuracy parameter to optimize in grid
                search, acceptable strings: "recall", "precision", "f1-score"
            criterion_output - classification accuracy output node to optimize
                in grid search, acceptable ints: 0, 1
            orientation - number of orientation bins in HOG
            ppc - 2-tuple size of pixels of a cell in HOG
            cpb - 2-tuple number of cells in each block in HOG
        """
        if self.classifier is not None:
            print("Classifier already exists." +
                  "  Training procedure cancelled.\n")
            return

        trainer = EggTrainer(pos_img_dir, neg_img_dir, orientations, ppc, cpb)

        if optimize:
            c, gamma, kernel = trainer.grid_search()
            self.classifier = trainer.model_train(c=c, gamma=gamma,
                                                  kernel=kernel, tts=tt_split)

        else:
            self.classifier = trainer.model_train(tts=tt_split)

    def save_classifier(file_loc):
        pickle.dump(self.classifier, open(file_loc + "egg_clf_" + time.time()
                                          + ".p", "wb+"))
                                          
    def determine_ev(self, img_fn):
        """opens recently saved file from camera feed and sends it through classifier
        """
        img = io.imread(img_fn, as_grey=True)
        print("Creating histogram of oriented gradients for image...")
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True, 
                    block_norm="L2-Hys")
        print("Making a prediction...")
        prediction = self.classifier.predict([fd])
        print("Saving histogram to file...")
        plt.imsave("./hog_caps/{}_hog_{}.png".format(int(time.time()),
            prediction), hog_img)
        print("Save complete.")
        return prediction

    def poll_cams(self, parent_thread):

        self.vision = EggVision(2, 1)
        p_init_frame = None
        avg_frame = None

        # if 2nd cam, add self.vision.aux_cam.isOpened() here
        while (self.vision.primary_cam.isOpened() and self.kill_flag is False):

            # Press q to stop polling frames
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

            primary_frame, p_gray = self.vision.get_pri_cam_frame()
            p_gray = cv2.GaussianBlur(p_gray, (21, 21), 0)

            # Set text
            text = "Waiting for embryo\n{}".format(
                time.strftime("%a %b %d %Y    %I:%M:%S:{} %p".format(round(time.time() * 1000 % 1000, 2))))

            # When this is the first frame of the video
            if p_init_frame is None:
                p_init_frame = primary_frame
                continue

            if avg_frame is None:
                avg_frame = p_gray.copy().astype("float")
                continue

            #print("timer: {}\tsignal: {}".format(self.signal_timer,
             #   self.signal_sent))
                
            # reset the signal flags to be ready to send signals again
            if self.signal_sent is False:
                print("Throughput Routine Activated")
                self.tput_timer = None  # reset throughput timer if an embryo is actually seen
                self.signal_timer = None
                self.signal_sent = None

            if self.tput_timer is None:
                self.tput_timer = time.time()

            elif (time.time() - self.tput_timer) >= self.tput_delay and \
                    (self.signal_sent is None or self.signal_sent is False):

                # Run sorting routine into bad after a delay
                
                text = "No embryo detected, throughput routine activated."
                self.signal_sent = True
                serialHandler.SerialThread.send_to_box(parent_thread, "w")

            cv2.accumulateWeighted(p_gray, avg_frame, self.us_factor)
            frame_delta = cv2.absdiff(p_gray, cv2.convertScaleAbs(avg_frame))

            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            # check the contours
            for c in cnts:

                if cv2.contourArea(c) < self.min_area \
                        or time.time() - self.startup_time < self.startup_delay:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(primary_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if self.signal_timer is None:

                    self.signal_timer = time.time()
                    self.signal_sent = True
                    detection_time = time.time() * 1000
                    print(str(detection_time/1000) + "\t[INFO]\tIncoming object")
                    
                    img_fn = "CAPTURED_IMAGE_{}.jpg".format(int(time.time()))
                    rescaled_img = cv2.resize(primary_frame, (640, 480))
                    cv2.imwrite(img_fn, rescaled_img)
                    
                    # determine if motion detected corresponds to the
                    # detection of a viable (good) embryo
                    prediction = self.determine_ev(img_fn)
                    signal_output_time = time.time() * 1000
                    if prediction == 1:
                        # Insert signal sending function call here
                        print("Predicted good embryo")
                        serialHandler.SerialThread.send_to_box(parent_thread, "v")
                        
                    else:
                        print("Not a good embryo...")
                        serialHandler.SerialThread.send_to_box(parent_thread, "w")

                    print("Sent signal within {} milliseconds of detection.".format(
                        signal_output_time - detection_time))

                elif self.signal_sent is True:
                    # print("[DEBUG]    {} seconds since signal sent".format(
                    # time.time() - signal_timer))
                    pass

                # if text != "Object detected!":
                # print("[INFO]    Incoming object")
                text = "Object detected at {}!".format(detection_time / 1000)

            # add text to MD cam frame
            cv2.putText(primary_frame, "Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Display the resulting frame
            cv2.imshow('Feature Detection View', primary_frame)

        if self.kill_flag is True:
            cv2.destroyAllWindows()
            self.vision.close_cams()

        def run():
            pass


class EggTrainer(object):
    """EggTrainer will train an SVM on an image dataset and pickle the model

    SVM is imported from scikit-learn.  If optimize=True, the model will
    run a grid search algorithm to obtain optimal training results,
    prioritizing recall, precision, or f1-score on either positive or
    negative embryo results.

    Attributes:
      dataset - matrix of feature vectors in col 1 and labels in col 2
    """

    def __init__(self, pos_img_dir, neg_img_dir, orientations, ppc, cpb):
        super(EggTrainer, self).__init__()
        self.dataset = _collect_dataset(pos_img_dir, neg_img_dir,
                                        orientations, ppc, cpb)

    def __str__(self):
        return "Egg Trainer"

    def _collect_dataset(pos_dir, neg_dir, ori, ppc, cpb):

        good_img_fns = glob.glob(pos_dir + "*.png")
        bad_img_fns = glob.glob(neg_dir + "*.png")
        dataset = []

        print("Reading and calculating HOG of positives...")
        for i, img_fn in enumerate(good_img_fns):
            sys.stdout.write("\r%d%%" % (i * 100 / len(good_img_fns)))
            sys.stdout.flush()

            img = io.imread(img_fn, as_grey=True)
            fd = hog(img, orientations=ori, pixels_per_cell=ppc,
                     cells_per_block=cpb, visualise=False, block_norm="L2-Hys")
            dataset.append([fd, 1])

        print("Reading and calculating HOG of bad images...")
        for i, img_fn in enumerate(bad_img_fns):
            sys.stdout.write("\r%d%%" % (i * 100 / len(bad_img_fns)))
            sys.stdout.flush()

            img = io.imread(img_fn, as_grey=True)
            fd = hog(img, orientations=ori, pixels_per_cell=ppc,
                     cells_per_block=cpb, visualise=False, block_norm="L2-Hys")
            dataset.append([fd, 0])

        print("Generating feature vectors...")
        random.shuffle(dataset)
        return dataset

    def grid_search():
        # TODO: implement
        pass

    def model_train(c=0.1, gamma=0.01, kernel="poly", tts=0.8):
        # Trains SVM model to given parameters

        clf = None

        if kernel == "linear":
            clf = svm.LinearSVC(C=c)

        else:
            clf = svm.SVC(C=c, gamma=gamma, kernel=kernel)

        X = []
        y = []

        for i in self.dataset:
            X.append(i[0])
            y.append(i[1])

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts,
                                                            random_state=0)

        clf.fit(X_train, y_train)
        print("Performing Support Vector Machine performance test...")
        expected = y_test
        predicted = classifier.predict(X_test)

        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))

        return clf


class EggVision(object):
    """EggVision utilizes OpenCV to open and return camera feed

    Contains attributes parameterizing resolution, motion detection
    sensitivity, etc.

    Attributes:
        (none)
    """

    def __init__(self, pri_cam_idx, aux_cam_idx, pc_res_h=480, pc_res_w=640,
                 pc_fps=60, ac_res_h=480, ac_res_w=640, ac_fps=60):
        super(EggVision, self).__init__()

        # open cameras
        self.primary_cam = cv2.VideoCapture(self._get_pri_cam_id())
        print("camera feed opened")
        #self.aux_cam = cv2.VideoCapture(aux_cam_idx)

        # set resolution
        self.primary_cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, pc_res_w)
        self.primary_cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, pc_res_h)
        #self.aux_cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, ac_res_w)
        #self.aux_cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, ac_res_h)

        # set fps
        self.primary_cam.set(cv2.cv.CV_CAP_PROP_FPS, pc_fps)
        #self.aux_cam.set(cv2.cv.CV_CAP_PROP_FPS, ac_fps)

        self.text = "N/A"

    def _get_cam_frame(self, pri_cam=True):

        frame = None
        ret = True

        if pri_cam:
            ret, frame = self.primary_cam.read()

        else:
            print("Error reading primary cam frame")
            #ret, frame = self.aux_cam.read()

        if ret is not True:
            raise IOError("Error retrieving video frame.")

        # return the image and its greyscale
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    def _get_pri_cam_id(self):
        
        device_re = re.compile("E:\sID_SERIAL_SHORT=+(?P<serial>\w+)$", re.I)
        dev_search_range = 2
        found_devices = []
        pri_cam_serial = "3C139807"
        
        # find all webcam device serials
        for i in range(1, dev_search_range + 1):
            
            try:
                found_devices.append(subprocess.check_output("udevadm info --name /dev/video{} | grep \"SERIAL_SHORT\"".format(i), shell=True))
        
            except subprocess.CalledProcessError as e:
                print("{}, trying next index...")
        
        # look for a match and obtain the id
        for dev in found_devices:
            parsed_rep = dev.split("\n")
            
            if parsed_rep:
                dev_serial = device_re.match(str(parsed_rep[0]))
                if dev_serial:
                    cam_serial = dev_serial.groupdict()
                    if cam_serial["serial"] == pri_cam_serial:
                        print("found primary camera, serial={}".format(pri_cam_serial))
                        return i
                
        print("WARNING: No the primary camera was not found, defaulting to idx 0: behavior unpredictable...")
        return 0
        

    def get_pri_cam_frame(self):
        return self._get_cam_frame()

    def get_aux_cam_frame(self):
        return self._get_cam_frame()

    def close_cams(self):
        self.primary_cam.release()
        #self.aux_cam.release()

    def __str__(self):
        return ("Egg Vision")

class EggTalker(object):
    """EggTalker opens USB communication with a device and sends model results

    Default USB device is the RedStick

    Attributes:
        device_name - the name of the usb port connected to the RedStick
        baud_rate - data transfer rate of serial communication
        timeout - communication timeout
        ser_dev - Serial USB device object
    """

    # Constants
    P_ON_CMD_CHAR = "g"
    P_OFF_CMD_CHAR = "h"
    C_ON_CMD_CHAR = "e"
    C_OFF_CMD_CHAR = "f"
    TC_ON_CMD_CHAR = "c"
    TC_OFF_CMD_CHAR = "d"

    def __init__(self, device_name, baud_rate=57600, timeout=1):
        super(EggTalker, self).__init__()
        self.device_name = device_name
        self.baud_rate = baud_rate
        self.timeout = timeout

    def __str__(self):
        return ("Egg Talker")

    """def __enter__(self):
        self._open_comms()

    def __exit__(self, exc_type, exc_value, traceback):
        self._close_comms()"""

    def open_comms(self):

        self.ser_dev = serial.Serial(self.device_name, self.baud_rate,
                                     timeout=self.timeout)

        if self.ser_dev.is_open() is False:
            raise IOError("Error opening serial device.")

    def close_comms(self):
        self.ser_dev.close()

    def _send_dev_cmd(self, cmd, verify):

        if self.ser_dev is None:
            raise IOError("No device open.")

        self.ser_dev.write(cmd)
        """if verify not in self.ser_dev.readline():
        raise IOError("Command could not be verified.")"""

    def turn_pump_off(self):
        self._send_dev_cmd(P_OFF_CMD_CHAR, P_OFF_CMD_VERIFY)

    def turn_pump_on(self):
        self._send_dev_cmd(P_ON_CMD_CHAR, P_ON_CMD_VERIFY)

    def turn_clamp_on(self):
        self._send_dev_cmd(C_ON_CMD_CHAR, C_ON_CMD_VERIFY)

    def turn_clamp_off(self):
        self._send_dev_cmd(C_OFF_CMD_CHAR, C_OFF_CMD_VERIFY)

    def turn_tri_clamp_on(self):
        self._send_dev_cmd(TC_ON_CMD_CHAR, TC_ON_CMD_VERIFY)

    def turn_tri_clamp_off(self):
        self._send_dev_cmd(TC_OFF_CMD_CHAR, TC_OFF_CMD_VERIFY)
