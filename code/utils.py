"""
This file will contain helper methods for use across files
"""
import csv
import heapq
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, mpld3
import pickle
import seaborn as sb
import time

import constants

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import SVC


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
    scatter = ax.scatter(tsne_embs[:,0], tsne_embs[:, 1], s=30)
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

def get_nearest_neighbors_emails(data, w2v, nb_size=3):
    """
    gets the nb_size nearest neighbors for any random email from data.
    distance is calculated based on embeddings obtained from w2v
    :param data:
    :param w2v:
    :param nb_size:
    :return:
    """
    h = []
    anchor_row = np.random.choice(len(data), 1)[0]
    anchor_word_embs = w2v.get_sentence(data[anchor_row, constants.EMAIL_BODY])
    if len(anchor_word_embs) == 0:
        print 'random anchor embedding incorrect, please run the program again'
        return
    anchor_emb = np.mean(anchor_word_embs, axis=0)
    minD = 10000000000
    for i in range(len(data)):
        if i == anchor_row: continue
        word_embs = w2v.get_sentence(data[i, constants.EMAIL_BODY])
        if len(word_embs) == 0: continue
        emb = np.mean(word_embs, axis=0)
        # since heapq is a min heap we insert negative of distance
        d = -np.sum((emb-anchor_emb)*(emb-anchor_emb))
        minD = min(-d, minD)
        # insert if the heap is not full yet or if d is greater than min element
        if len(h) <= nb_size or d > h[0][0]:
            heapq.heappush(h, (d, data[i, constants.EMAIL_BODY]))
        if len(h) > nb_size:
            heapq.heappop(h)
    print 'minimum distance found ', minD
    print 'Anchor Email:'
    print data[anchor_row, constants.EMAIL_BODY]
    print '\n'
    print 'Nearest neighbors in order'
    for i in range(len(h)):
        print 'Email ' + str(i+1) + ' with distance ' + str(-h[i][0])
        print h[i][1]
        print '\n'


def save_user_embeddings(email_ids, embeddings):
    """
    the method saves the email embeddings passed to it as a pickle file
    :return: saves as pickle file
    """
    filepath = '../resources/' + 'embeddings_' + constants.RUN_ID + '.pkl'
    embs = (email_ids, embeddings)
    pickle.dump(embs, open(filepath, "wb"))


def load_user_embeddings():
    """
    The method loads user embeddings from a pickle file and returns a list of emails and a list of embeddings
    :return:
    """
    filepath = '../resources/embeddings.pkl'
    email_ids, embeddings = pickle.load(open(filepath, "rb"))
    return email_ids, embeddings



def get_similar_users(labels, embeddings, nb_size=3):
    for i in range(embeddings.shape[0]):
        anchor_emb = embeddings[i]
        h = []
        minD = 10000000000
        for j in range(embeddings.shape[0]):
            if i == j: continue
            emb = embeddings[j]
            d = -np.sum((anchor_emb-emb)*(anchor_emb-emb))
            minD = min(-d, minD)
            # insert if the heap is not full yet or if d is greater than min element
            if len(h) <= nb_size or d > h[0][0]:
                heapq.heappush(h, (d, labels[j]))
            if len(h) > nb_size:
                heapq.heappop(h)
        print '\n'
        print 'anchor person: ', labels[i]
        print 'minimum distance found', minD
        print 'nearest neighbors:'
        for k in range(len(h)):
            print h[k][1], ' with distance=', (-h[k][0])


def load_user_designations():
    """
    the method reads the csv file containing user info and returns a list of email_ids along with their designation
    """
    filepath = '../resources/employee_info.csv'
    with open(filepath, 'rb') as f:
        reader = csv.reader(f)
        designations = {}
        for row in reader:
            if row[-1] == 'status' or row[-1] == 'NULL' or row[-1] == 'N/A':
                continue
            designations[row[2]] = row[-1]
    # return designations
    return desgn_categorization1(designations)


def desgn_categorization1(designations):
    for key,val in designations.iteritems():
        if val == 'CEO' or val == 'President' or val == 'Vice President':
            designations[key] = 'CEO/Presidents'
        elif val == 'Managing Director' or val == 'Director':
            designations[key] = 'Directors'
        elif val == 'Manager':
            designations[key] = 'Manager'
        elif val == 'Employee':
            designations[key] = 'Employee'
        else:
            designations[key] = 'Others'
    return designations


def desgn_categorization2(designations):
    for key,val in designations.iteritems():
        if val == 'CEO' or val == 'President' or val == 'Vice President':
            designations[key] = 'Upper Management'
        elif val == 'Managing Director' or val == 'Director' or val == 'Manager':
            designations[key] = 'Middle Management'
        elif val == 'Employee':
            designations[key] = 'Employee'
        else:
            designations[key] = 'Others'
    return designations


def extract_emb_desgn():
    designations = load_user_designations()
    emailids, embs = load_user_embeddings()
    # remove email embeddings that don't have a designation
    X = []
    y=[]
    for i in range(len(emailids)):
        e_id = emailids[i]
        if e_id in designations:
            X.append(embs[i])
            y.append(designations[e_id])
    return np.array(X), np.array(y)


def k_fold_cross_validation():
    # extract the data
    X, y = extract_emb_desgn()
    # split the data into k-folds
    kf = KFold(n_splits=23, shuffle=True)
    # run k-fold cross validation
    cor = 0
    y_t = np.array([])
    y_p = np.array([])
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        classifier = SVC()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print y_pred
        cor = cor + np.sum(y_pred == y_test)
        y_t = np.append(y_t, y_test)
        y_p = np.append(y_p, y_pred)
    print (cor * 1.0) / len(y)

    #plot the confusion matrix
    confusion_matrix(y_t, y_p)
