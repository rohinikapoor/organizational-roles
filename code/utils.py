"""
This file will contain helper methods for use across files
"""

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
