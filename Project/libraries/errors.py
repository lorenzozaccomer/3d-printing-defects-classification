###
# 
#  errors.py
#
###

import os
import logging

def CheckParametersErrors(EPOCHS, LEARNING_RATE, ITERATION,
     VISUALIZE_PREDICTION, TEST_IMAGE_PATH, MODEL_PATH):
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
    else:
        print("not error, you can proceed")
        logging.info("NOT ERROR ON YOUR PARAMETER")


