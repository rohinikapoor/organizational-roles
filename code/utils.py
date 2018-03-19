"""
This file will contain helper methods for use across files
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, mpld3
import time
import seaborn as sb


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


def plot_with_tsne(labels, embeddings, display_hover=True):
    """
    expects a list of email_ids and numpy ndarray of embeddings. The numpy ndarray should have shape L,D where D is the
    size of embeddings and L is the number of users
    """
    tsne = TSNE(verbose=1, method='exact')
    start = time.time()
    tsne_embs = tsne.fit_transform(embeddings)
    end = time.time()
    print('time taken by TSNE ', (end-start))

    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_embs[:,0], tsne_embs[:,1], s=30)
    if display_hover:
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
        mpld3.plugins.connect(fig, tooltip)
        mpld3.show()
    else:
        plt.savefig('../outputs/tsne-users.png')


def plot_emails_with_tsne(email_data, w2v, display_hover=True):
    """
    below code generates embedding for each email (using w2v model) and plots the embedding using unique sender color
    """
    sb_color_list = sb.xkcd_rgb.values()
    color_pos = 0
    sender2color_map = {}
    embeddings = []
    plot_color_list = []
    plot_senders = []
    for i in range(len(email_data)):
        sender = email_data[i, 0]
        if get_userid(sender) is None:
            continue
        if sender in sender2color_map:
            sender_color = sender2color_map[sender]
        else:
            sender2color_map[sender] = sb_color_list[color_pos]
            sender_color = sb_color_list[color_pos]
            color_pos += 1
        email_words_emb = w2v.get_sentence(email_data[i, 2])
        if len(email_words_emb) == 0:
            continue
        embeddings.append(np.mean(email_words_emb, axis=0))
        plot_color_list.append(sender_color)
        plot_senders.append(sender)

    # run tsne, the number of emails can get very large ~70k
    tsne = TSNE(verbose=1)
    start = time.time()
    tsne_embs = tsne.fit_transform(embeddings)
    end = time.time()
    print('time taken by TSNE ', (end - start))

    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_embs[:, 0], tsne_embs[:, 1], c=plot_color_list, s=70)
    if display_hover:
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=plot_senders)
        mpld3.plugins.connect(fig, tooltip)
        mpld3.show()
    else:
        plt.savefig('../outputs/tsne-emails.png')

