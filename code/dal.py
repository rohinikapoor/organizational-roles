import csv
import MySQLdb
import numpy as np
import os
import random
import re
import time
import utils

# from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize.stanford import CoreNLPTokenizer


def get_emails(num_emails=100, fetch_all=False):
    """ Returns requested number of emails from a csv file. The function returns a numpy array of dimension (N,3)
    where N is the number of rows, dimension1 is senderId, dimension2 is receiverIds, dimension3 is message body
    If there are multiple csv files present, it selects the appropriate file that meets the <num_emails> requirement
    and returns the requested number of rows. Use this method to get a fixed number of mails.
    :param num_emails: The number of rows to be returned
    :param fetch_all: Ignores num_emails & fetches all the emails in the enron dataset. The number of returned rows may
    be very large """
    file_path = '../data/'
    print 'Loading data'
    if fetch_all:
        file_name = 'all_emails.csv'
        f = open(file_path+file_name, 'r')
    else:
        file_name = __get_appr_filename(num_emails, file_path)
        f = open(file_path+file_name, 'rb')
    
    with f:
        reader = csv.reader(f)
        data = list(reader)

    if not fetch_all and len(data) > num_emails:
        data = data[:num_emails]
    print 'Data loaded'
    # utils.plot_email_date_distribution(data)
    return np.array(data)


def get_emails_by_users(num_users=150):
    """
    Randomly selects num_users from the 150 users list and trims all_emails to return the mails only for selected
    num users. Use this method to get all the data for fixed number of users
    :param num_users:
    :return:
    """
    print 'Loading data'
    filepath = '../data/all_emails.csv'
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = __filter_mails_by_users(data, num_users)
    print 'Data loaded'
    return np.array(data)


def dataset_split(data, val_split=0.0, test_split=0.3):
    train_split = int((1 - val_split - test_split) * len(data))
    val_split = train_split + int(val_split * len(data))

    data = np.array(sorted(data, key=lambda x: x[3]))
    train_data = data[: train_split]
    val_data = data[train_split: val_split]
    test_data = data[val_split:]

    return train_data, val_data, test_data


def dataset_filter_by_user(data, val_split=0.0, test_split=0.3, threshold=1):
    train, val, test = [], [], []
    mails_by_senders = utils.group_mails_by_sender(data)
    for sender in mails_by_senders:
        mails = mails_by_senders[sender]

        train_split = int((1 - val_split - test_split) * len(mails))
        val_split = train_split + int(val_split * len(mails))

        mails = np.array(sorted(mails, key=lambda x: x[3]))
        train.append(mails[: train_split])

        if train_split > threshold:
            val.append(mails[train_split: val_split])
            test.append(mails[val_split:])

    return np.concatenate(train), np.concatenate(val), np.concatenate(test)


def get_negative_emails(emails, fraction=1.0):
    """
    Get examples of mails not sent by sender and receiver pairs as negative examples for classification
    :param emails:
    :param fraction:
    :return:
    """
    # TODO: Discuss the implications of using external mails instead of restricting ourselves to mails to/fro 150 core
    # TODO: Discuss the implications of using similar recipients and number of recipients
    print 'Generating negative examples'
    filepath = '../data/all_emails.csv'
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    num_examples = min(int(len(emails) * fraction), len(data))
    random_state = random.getstate()
    random.seed(47)
    random.shuffle(data)
    data = data[:num_examples]
    i = 0
    while i < num_examples:
        rand_idx = random.randrange(0, len(emails))
        if data[i][0] != emails[rand_idx][0]:
            data[i][0] = emails[rand_idx][0]
            data[i][1] = emails[rand_idx][1]
            # TODO: See if date is relevant
            data[i][3] = emails[rand_idx][3]
            i += 1
    random.setstate(random_state)

    print 'Examples created'
    return np.array(data)


def __filter_mails_by_users(emails, max_users):
    print 'Filtering mail by users'
    email_ids = utils.get_user_emails()
    max_users = min(max_users, len(email_ids))
    
    # To have deterministic behaviour over number of users
    # While having a knob to tweak which users get picked (and thus, number of emails)
    random_state = random.getstate()
    random.seed(42)
    filtered_ids = set(random.sample(email_ids, max_users))
    random.setstate(random_state)
    
    # We retain only those mails that have both valid senders and atleast one valid receiver
    filtered_mails = []
    for sender, receivers, mail, date in emails:
        if sender in filtered_ids:
            for receiver in receivers.split('|'):
                if receiver in filtered_ids:
                    filtered_mails.append((sender, receivers, mail, date))
                    break

    return filtered_mails


def load_from_db(num_emails=100, fetch_all=False):
    """loads the requested number of emails from the database and dumps the file as csv. For this function to
    run the mysql database must be set up locally.
    if fetch_all is false, file is stored as <num_emails>_emails.csv
    if the fetch_all is true, file is stored as all_emails.csv
    :param num_emails: The number of emails to be fetched from the db
    :param fetch_all: Will fetch all the emails from db. Large join query that may take some time"""
    db_conn = MySQLdb.connect(host='localhost', user='root', passwd='mudit1990', db='EnronAHS')
    file_path = '../data/'
    if fetch_all:
        res = __db_query_all(db_conn)
        file_name = 'all_emails.csv'
    else:
        file_name = str(num_emails) + '_emails.csv'
        res = __db_query_partial(db_conn, num_emails)
    data = __clean_data_glove(res)
    with open(file_path+file_name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def __get_appr_filename(num_emails, file_path):
    """
    gets the appropriate csv file based on the num_emails required. Tries to select the file with minimum rows that
    meets the requirement
    :param num_emails:
    :param file_path:
    :return:
    """
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    max_rows = -1
    rec_rows = -1
    for f in files:
        f_list = f.split('_')
        try:
            rows = int(f_list[0])
            if rows > max_rows:
                max_rows = rows
            if rows >= num_emails and (rows < rec_rows or rec_rows == -1):
                rec_rows = rows
        except ValueError:
            continue
    if rec_rows == -1 and max_rows == -1:
        return None
    if rec_rows == -1:
        rec_rows = max_rows
    return str(rec_rows) + '_emails.csv'


def __clean_data_glove(data):
    """
    The function assumes input as a tuple of tuples as returned from db , cleans the data and returns a list of list
    The following cleaning steps are performed
    1) multiple receivers are separated by '|'
    2) all the strings are converted into lowercase
    3) email body is cleaned using stanfordtokenizer. It tokenizes the scentences into words. Punctuations are separated
    and considered as individual words. This is compatible with word2vec glove model which makes use of the same
    tokenizer
    """
    # st = StanfordTokenizer(path_to_jar='../resources/stanford-corenlp-3.9.1.jar')
    st = CoreNLPTokenizer()
    clean_mail = lambda x: (' '.join(st.tokenize(x))).encode('ascii', 'ignore')
    cleaned_data = []
    for i, row in enumerate(data):
        if i % 1000 == 0:
            print 100 * (i + 0.0) / len(data), '% emails processed'
        try:
            cleaned_row = list(row)
            # replace ',' separator in receivers with '|'
            cleaned_row[2] = cleaned_row[2].replace(',', '|')
            # convert the email body to lower case
            cleaned_row[3] = cleaned_row[3].lower()
            # put space after full stops since nltk can't separate those
            cleaned_row[3] = re.sub(r'\.(?=[^ \W\d])', '. ', cleaned_row[3])
            # use nltk stanford tokenizer to clean the email body
            cleaned_mail_thread = clean_mail(cleaned_row[3])
            cleaned_row[3] = __truncate_email(cleaned_mail_thread)
            # remove the first random id column and append ot to cleaned_data
            cleaned_data.append(cleaned_row[1:])
        except Exception as e:
            print i, row, e

    return cleaned_data


def __truncate_email(em):
    """
    truncates the email after first occurrence of either
    1) original message
    2) forwarded by
    """
    eot = len(em)
    # find original message
    eot1 = em.find('original message')
    if eot1 >= 0:
        eot = min(eot, eot1)
    # find forwarded by
    # eot2 = em.find('forwarded by')
    # if eot2 >= 0:
    #     eot = min(eot, eot2)
    return em[:eot]


def __db_query_all(db_conn):
    """
    :param db_conn:
    :return: tuples of tuples containing data
    """
    query = "select message.mid, sender, group_concat(rvalue) as receivers, body, message.date " \
            "from EnronAHS.message " \
            "inner join EnronAHS.recipientinfo " \
            "on message.mid=recipientinfo.mid " \
            "group by message.mid;"
    cur = db_conn.cursor()
    cur.execute("SET SESSION group_concat_max_len = 100000;")
    cur.execute(query)
    return cur.fetchall()


def __db_query_partial(db_conn, num_emails):
    """
    :param db_conn:
    :param num_emails:
    :return tuple of tuples containing the data
    """
    query = "select m.mid, m.sender, group_concat(rvalue) as receivers, m.body, m.date from" + \
            "(select * from EnronAHS.message order by rand() limit " + str(num_emails) + \
            ")  as m inner join EnronAHS.recipientinfo on m.mid=recipientinfo.mid group by m.mid;"
    cur = db_conn.cursor()
    cur.execute("SET SESSION group_concat_max_len = 100000;")
    cur.execute(query)
    return cur.fetchall()
