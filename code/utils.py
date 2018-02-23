"""
This file will contain helper methods for use across files
"""
import numpy as np

# a dictionary that stores the mapping of email_id to id that is used to lookup user embeddings
user_id_lookup = {}


def get_userid(u):
    """
    gets the unique user_id for every user. Returns none if the mapping is not present
    :param u:
    :return:
    """
    if u in user_id_lookup:
        return user_id_lookup[u]
    else:
        return None


def populate_userid_mapping():
    mapping = np.loadtxt('../resources/employee_id_mapping.csv', dtype='str', delimiter=',', skiprows=1)
    for m in mapping:
        user_id_lookup[m[0]] = int(m[1])
    print user_id_lookup


