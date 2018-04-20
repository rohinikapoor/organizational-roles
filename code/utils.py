"""
This file will contain helper methods for use across files
"""
import csv
from datetime import datetime
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
from sklearn.model_selection import train_test_split


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


def assign_labels_colors(labels, colors):
    """
    Takes a list of labels and colors and assigns a unique label to each color. Returns a color_list of length(labels).
    The colors will loop around if the number of unique labels are more than the number of unique colors
    :param labels:
    :param colors:
    :return: color_list
    """
    col_idx = 0
    label2col = {}
    col_list = []
    for i in range(len(labels)):
        if labels[i] in label2col:
            col_list.append(label2col[labels[i]])
        else:
            col = colors[col_idx%len(colors)]
            col_idx += 1
            label2col[labels[i]] = col
            col_list.append(col)
    return col_list


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

    # creating the colors
    colors = list(sb.color_palette().as_hex())
    color_list = assign_labels_colors(labels, colors)

    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_embs[:,0], tsne_embs[:, 1], c=color_list, s=75)
    if display_hover:
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
        mpld3.plugins.connect(fig, tooltip)
        mpld3.show()
    else:
        outfile = '../outputs/' + constants.RUN_ID + '_tsne-users.png'
        plt.savefig(outfile)


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


def plot_email_date_distribution(email_data):
    list_of_datetimes = [datetime.strptime(x[3], '%Y-%m-%d %H:%M:%S') for x in email_data if '1999' < x[3] < '2003']
    list_of_datetimes.sort()
    dates = mpl.dates.date2num(list_of_datetimes)
    plt.xticks(rotation=60)
    plt.plot()
    plt.plot_date(dates, np.arange(0.0, len(dates)) / len(dates), '--')
    plt.savefig('../outputs/email-date-distribution.png', bbox_inches='tight')


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


def load_user_embeddings(filepath):
    """
    The method loads user embeddings from a pickle file and returns a list of emails and a list of embeddings
    :return:
    """
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


def load_user_designations(cat):
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
    if cat == 'cat1':
        return desgn_categorization1(designations)
    elif cat == 'cat2':
        return desgn_categorization2(designations)
    else:
        return designations


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


def extract_emb_desgn(emailids, embs, cat='None'):
    designations = load_user_designations(cat)
    # remove email embeddings that don't have a designation
    X = []
    y=[]
    for i in range(len(emailids)):
        e_id = emailids[i]
        if e_id in designations:
            X.append(embs[i])
            y.append(designations[e_id])
    return np.array(X), np.array(y)


def get_dominance_data(email_ids, embs):
    """
    :param email_ids: Email Ids of Enron people corresponding to their embedding
    :param embs: Trained embeddings for people
    :return X: concatenated pair of embeddings emb1,emb2
    :return y: -1, 0 or 1 where -1 means desgn(emb1)<desgn(emb1), 0 when desgn(emb1)=desgn(emb1) &
     +1 when desgn(emb1)>desgn(emb1)
    """
    desgn_order = {'Employee': 0,
                   'Trader': 0,
                   'Manager': 1,
                   'In House Lawyer': 1,
                   'Managing Director': 2,
                   'Director': 3,
                   'Vice President': 4,
                   'President': 5,
                   'CEO': 6}
    user_desgn = load_user_designations(cat='None')
    dom_embs = []
    dom_y = []
    for i in range(len(email_ids)):
        for j in range(i+1, len(email_ids)):
            if email_ids[i] not in user_desgn or email_ids[j] not in user_desgn or i == j:
                continue
            user_dg_i = user_desgn[email_ids[i]]
            user_dg_j = user_desgn[email_ids[j]]
            p = np.random.rand()
            if p > 0.5:
                pair_emb, y = get_dominance_relation(user_dg_i, user_dg_j, embs[i], embs[j], desgn_order)
            else:
                pair_emb, y = get_dominance_relation(user_dg_j, user_dg_i, embs[j], embs[i], desgn_order)
            dom_embs.append(pair_emb)
            dom_y.append(y)
    return np.array(dom_embs), np.array(dom_y)


def get_dominance_relation(dg1, dg2, emb1, emb2, desgn_order):
    pair_emb = np.concatenate((emb1, emb2))
    if desgn_order[dg1] < desgn_order[dg2]:
        y = -1
    elif desgn_order[dg1] > desgn_order[dg2]:
        y = 1
    else:
        y = 0
    return pair_emb, y


def group_mails_by_sender(emails):
    dataset = {}
    for sender, receivers, email, date in emails:
        if sender not in dataset:
            dataset[sender] = list()
        dataset[sender].append((sender, receivers, email, date))
    for sender in dataset:
        dataset[sender] = np.array(dataset[sender])
    return dataset


def split_by_users(email_ids, embs):
    tr, te = get_user_split(email_ids)
    train_set_usr, test_set_usr = set(tr), set(te)
    train_users, train_embs, test_users, test_embs = [], [], [], []
    for i in range(len(email_ids)):
        if email_ids[i] in train_set_usr:
            train_users.append(email_ids[i])
            train_embs.append(embs[i])
        elif email_ids[i] in test_set_usr:
            test_users.append(email_ids[i])
            test_embs.append(embs[i])
    # print np.array(train_users).shape, np.array(train_embs).shape, np.array(test_users).shape, np.array(test_embs).shape
    return train_users, train_embs, test_users, test_embs


def get_user_split(email_ids):
    email_ids_set = set(email_ids)
    designations = load_user_designations(cat='None')
    desgn_set = set(designations.keys())
    filt_email_ids = list(email_ids_set.intersection(desgn_set))
    train_users, test_users = train_test_split(filt_email_ids, test_size=0.20, random_state=42, shuffle=True)
    # verify_user_split(train_users, test_users)
    return train_users, test_users


def verify_user_split(train_users, test_users):
    em_freq = get_emailid_freq()
    train_frq = [em_freq[em] for em in train_users]
    test_frq = [em_freq[em] for em in test_users]
    print len(test_frq), len(train_frq)
    print 'test min:', np.min(test_frq)
    print 'test max:', np.max(test_frq)
    print 'test mean:', np.mean(test_frq)
    print 'test std:', np.std(test_frq)
    print 'test sum', np.sum(test_frq)
    print 'train min:', np.min(train_frq)
    print 'train max:', np.max(train_frq)
    print 'train mean:', np.mean(train_frq)
    print 'train std:', np.std(train_frq)
    print 'train_sum', np.sum(train_frq)


def get_emailid_freq():
    with open('../resources/emailid_mailfreq.pkl', 'rb') as f:
        em_freq = pickle.load(f)
    return em_freq

# em_freq = get_emailid_freq()
# email_ids, embs = load_user_embeddings('../important_embeddings/usr50d_em50d_m2faster_20ep/embeddings_usr50d_em50d_m2faster_20ep.pkl')
# designations = load_user_designations(cat='None')
# email_set = set(email_ids)
# desgn_set = set(designations.keys())
# email_set = email_set.intersection(desgn_set)
# split_by_users(email_ids, embs)
# get_user_split(list(email_set), em_freq)
# x, y = get_dominance_data(email_ids, embs)
# k_fold_cross_validation(email_ids, embs)
# email_ids, embs = load_user_embeddings('../important_embeddings/embeddings_usr50d_em100d_custom_25ep_m3/embeddings_usr50d_em100d_custom_25ep_m3.pkl')
# embs, labels = extract_emb_desgn(email_ids, embs, cat='cat1')
# plot_with_tsne(labels.tolist(), embs)
