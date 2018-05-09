import sys
"""
This file is used to define constants used throughout the code
"""
# ordering of the data columns in csv files
SENDER_EMAIL = 0
RECEIVER_EMAILS = 1
EMAIL_BODY = 2

# defining the dimensions for email, sender and receiver embedding
EMAIL_EMB_SIZE = int(sys.argv[8])
WORD_CORPUS_SIZE = sys.argv[6]
USER_EMB_SIZE = int(sys.argv[7])

# Commandline arguments
RUN_ID = sys.argv[1]

# constant paths w.r.t to code directory
MODEL_DIR = '../models/'

# configure the number of hidden layers
HIDDEN_DIMS = eval(sys.argv[5])