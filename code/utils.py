"""
This file will contain helper methods for use across files
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt, mpld3
import time


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


def get_user_emails():
    return user_id_lookup.keys()


def populate_userid_mapping():
    """
    populates the user_id_lookup table based on file employee_id_mapping.csv extracted from the db
    """
    mapping = np.loadtxt('../resources/employee_id_mapping.csv', dtype='str', delimiter=',', skiprows=1)
    for m in mapping:
        user_id_lookup[m[0]] = int(m[1])


def plot_with_tsne(email_ids, embeddings):
    """
    expects a list of email_ids and numpy ndarray of embeddings. The numpy ndarray should have shape L,D where D is the
    size of embeddings and L is the number of users
    :param email2embedding:
    :return:
    """
    tsne = TSNE(verbose=1, method='exact')
    start = time.time()
    tsne_embs = tsne.fit_transform(embeddings)
    end = time.time()
    print('time taken by TSNE ', (end-start))

    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_embs[:,0], tsne_embs[:,1], s=30)
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=email_ids)
    mpld3.plugins.connect(fig, tooltip)
    mpld3.show()


