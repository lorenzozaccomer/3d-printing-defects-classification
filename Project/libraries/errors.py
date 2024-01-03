###
# 
#  errors.py
#
###

import os
import logging

from model import *

def CheckParametersErrors(EPOCHS, LEARNING_RATE, ITERATION,
     VISUALIZE_PREDICTION, TEST_IMAGE_PATH, MODEL_PATH, DATASET_PATH):
    """
    This function check if the user set wrong paramater, in this case
    the script will call the exit() system function and the script will close
    """

    print("check paramaters ...")
    logging.info("CHECK PARAMETERS")

    if EPOCHS <= 0:
        print("epochs error! Change the value")
        logging.debug("EPOCHS ERROR")
        exit()
    elif LEARNING_RATE < 0:
        print("learning rate error! Change the value")
        logging.debug("LEARNING_RATE ERROR")
        exit()
    elif ITERATION not in [0,1]:
        print("iteration value error! Change the value")
        logging.debug("ITERATION ERROR")
        exit()
    elif VISUALIZE_PREDICTION not in [0,1]:
        print("visualize prediction value error! Change the value")
        logging.debug("VISUALIZE_PREDICTION ERROR")
        exit()
    elif not(os.path.exists(TEST_IMAGE_PATH)):
        print("test image path error! this image file doesn't exist")
        logging.debug("TEST_IMAGE_PATH ERROR")
        exit()
    elif not(os.path.exists(MODEL_PATH)):
        print("model path error! this image file doesn't exist")
        logging.debug("MODEL_PATH ERROR")
        exit()
    elif not(os.path.isdir(DATASET_PATH)):
        print("dataset image path error! it not exists")
        logging.debug("DATASET_PATH ERROR")
        exit()
    if not(os.listdir(DATASET_PATH)):
        print("On your path there aren't folders!")
        logging.debug("NOT DIRECTORIES ERROR")
        exit()
    else:
        print("not error, you can proceed")
        logging.info("NOT ERROR ON YOUR PARAMETER")

    
    logging.info("ITERATION: " + str(ITERATION))
    logging.info("VISUALIZE_PREDICTION: " + str(VISUALIZE_PREDICTION))
    logging.info("EPOCHS: " + str(EPOCHS))
    logging.info("TEST_IMAGE_PATH: " + TEST_IMAGE_PATH)
    logging.info("MODEL_PATH: " + MODEL_PATH)
    logging.info("DATASET_PATH: " + DATASET_PATH)
    logging.info("DATASET_SUB_DIRECTORIES: " + str(os.listdir(DATASET_PATH)))

    if ITERATION == 1:
        print('execution of model generation function')
        ModelGeneration(DATASET_PATH, MODEL_PATH, EPOCHS, LEARNING_RATE)
    elif ITERATION == 0 and VISUALIZE_PREDICTION == 1:
        print('skip model generation, it loads the model from path and visualize the results')
        LoadModelVisualization(DATASET_PATH, MODEL_PATH)
        exit()
    else:
        print("loading image classification .. ")


    logging.info("-----------   NEW IMAGE CLASSIFICATION  -----------")
    logging.info("loading image classification .. ")