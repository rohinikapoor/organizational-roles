import sys
"""
This file is used to define constants used throughout the code
"""
# ordering of the data columns in csv files
SENDER_EMAIL = 0
RECEIVER_EMAILS = 1
EMAIL_BODY = 2

# defining the dimensions for email, sender and receiver embedding
EMAIL_EMB_SIZE = 300
USER_EMB_SIZE = 100

# Commandline arguments
RUN_ID = sys.argv[1]