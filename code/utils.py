"""
This file will contain helper methods for use across files
"""
import numpy as np

# A dictionary that stores a mapping of unique_id to email_id. This unique_id is used to lookup the embeddings in
# nn.Embeddings layer
user_id_lookup = {}


def get_userid(u):
    """
    gets the unique_id based on the email. Returns None if the email is not present in the dictionary
    :param u:
    :return:
    """
    if u in user_id_lookup:
        return user_id_lookup[u]
    else:
        return None


def populate_userid_mapping():
    """

    """
    mapping = np.loadtxt('../resources/employee_id_mapping.csv', dtype='str', delimiter=',', skiprows=1)
    for m in mapping:
        user_id_lookup[m[0]] = int(m[1])


