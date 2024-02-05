====================================================
================== Project Setup ===================
====================================================

once you download the project, open CMD in the project directory or open CMD and use:
    cd PROJECT_DIRECTORY
run this command to install all the required packages:
    pip install -r requirements.txt


======================================================================
================== Gathering data for each emotion ===================
======================================================================

(python gather-data.py --help) to show description of the parameters

use these 7 commands to gather data for each of the 7 emotions
specify the number of iterations for each cmd
! it is a slow process if iterations are high !
!! emotion labels are case sensitive !!

    python gather-data.py --iterations 100 --emotion Angry
    python gather-data.py --iterations 100 --emotion Disgust
    python gather-data.py --iterations 100 --emotion Fear
    python gather-data.py --iterations 100 --emotion Happy
    python gather-data.py --iterations 100 --emotion Neutral
    python gather-data.py --iterations 100 --emotion Sad
    python gather-data.py --iterations 100 --emotion Surprise



==============================================================
================== Training your own model ===================
==============================================================

(python train.py --help) to show description of the parameters

you can train your own model using this command
it is advice to use more than 6000 rows in the dataset to have good results

    python train.py -e NUMBER_OF_EPOCHS
                    -bs BATCH_SIZE 
                    -lr LEARNING_RATE 
                    -ts TEST_SIZE
                    -dp DATASET_PATH
                    -mp OUTPUT_MODEL_PATH
                    -hd TRAINING_HISTORY_PATH

== OR ==

    python train.py --epochs NUMBER_OF_EPOCHS
                    --batch-size BATCH_SIZE 
                    --learning-rate LEARNING_RATE 
                    --test-split TEST_SIZE
                    --data-path DATASET_PATH
                    --model-path OUTPUT_MODEL_PATH
                    --history-dir TRAINING_HISTORY_PATH

=> recommended: USE THE DEFAULT TRAINING PARAMETERS USING:
    python train.py

=================================================
================== Test model ===================
=================================================

(python test-model.py --help) to show description of the parameters

test your model using the command below
since the data gathering process takes alot of time, you can test our model thats in ./model

    python test-model.py -mp MODEL_PATH

== OR == 

    python test-model.py --model-path MODEL_PATH

=> recommended: USE THE DEFAULT MODEL USING:
    python test-model.py
