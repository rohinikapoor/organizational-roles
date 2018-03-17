import MySQLdb
import numpy as np
import os
import random
import string
import time
import utils

from nltk.tokenize.stanford import StanfordTokenizer


def get_emails(num_emails=100, max_users=150, fetch_all=False):
    """ Returns requested number of emails from a csv file. The function returns a numpy array of dimension (N,3)
    where N is the number of rows, dimension1 is senderId, dimension2 is receiverIds, dimension3 is message body
    If there are multiple csv files present, it selects the appropriate file that meets the <num_emails> requirement
    and returns the requested number of rows
    :param num_emails: The number of rows to be returned
    :param fetch_all: Ignores num_emails & fetches all the emails in the enron dataset. The number of returned rows may
    be very large """
    file_path = '../data/'
    data = []
    if fetch_all:
        file_name = 'all_emails.csv'
        data = np.loadtxt(file_path + file_name, dtype='str', delimiter=',')
    else:
        file_name = __get_appr_filename(num_emails, file_path)
        data = np.loadtxt(file_path + file_name, dtype='str', delimiter=',')
    
    data = __filter_mails_by_users(data, max_users)
    if not fetch_all and len(data) > num_emails:
        data = data[:num_emails]
    return data

def __filter_mails_by_users(emails, max_users):
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
    for sender, receivers, mail in emails:
        if sender in filtered_ids:
            for receiver in receivers.split('|'):
                if receiver in filtered_ids:
                    filtered_mails.append((sender, receivers, mail))
                    break

    filtered_mails = np.array(filtered_mails)
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
    data = __clean_data(np.asarray(res))
    data = data[:, 1:] # remove the first column that contains the random id
    np.savetxt(file_path+file_name, data, fmt='%s', delimiter=',')


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


def __clean_data(data):
    """
    The function assumes an input as a numpy array, cleans the data and returns a numpy array
    """
    # replace ',' separator in receivers with '|'
    data[:, 2] = np.core.defchararray.replace(data[:, 2], ',', '|')
    # convert the email body to lower case
    data[:, 3] = np.core.defchararray.lower(data[:, 3])
    # delete all the punctuation marks from the email body
    data[:, 3] = np.core.defchararray.translate(data[:,3], None, string.punctuation)
    # replace '\n' with ' ' in email body
    data[:, 3] = np.core.defchararray.replace(data[:, 3], '\n', ' ')
    # replace '\r' with ' ' in email body
    data[:, 3] = np.core.defchararray.replace(data[:, 3], '\r', ' ')
    # replace '\t' with ' ' in email body
    data[:, 3] = np.core.defchararray.replace(data[:, 3], '\t', ' ')
    return data


def __clean_data_glove(data):
    """
    The function assumes an input as a numpy array, cleans the data and returns a numpy array
    This is meant to tokenize sentences into words that are in the GloVe vocabulary
    It does not remove punctuations as it is possible that they may provide additional information
    """
    # replace ',' separator in receivers with '|'
    data[:, 2] = np.core.defchararray.replace(data[:, 2], ',', '|')
    
    # convert the email body to lower case
    data[:, 3] = np.core.defchararray.lower(data[:, 3])
    
    # clean mails while retaining structure as required by GloVe
    st = StanfordTokenizer(path_to_jar='../resources/stanford-corenlp-3.9.1.jar')
    clean_mail = lambda x: ' '.join(st.tokenize(x))
    for idx in data.shape[0]:
        # Cannot be vectorized as ndarrays are not contiguous
        # TODO: See if we can refactor logic to avoid numpy arrays
        cleaned_mail_thread = clean_mail(data[idx, 3])
        
        # Truncate the email after the first mail in the thread
        end_of_thread = cleaned_mail_thread.find('original message')
        end_of_thread = len(cleaned_mail_thread) if end_of_thread == -1 else end_of_thread
        cleaned_mail = cleaned_mail_thread[:end_of_thread]
        data[idx, 3] = cleaned_mail

    return data

def __db_query_all(db_conn):
    """
    :param db_conn:
    :return: tuples of tuples containing data
    """
    query = "select message.mid, sender, group_concat(rvalue) as receivers, body from EnronAHS.message inner join EnronAHS.recipientinfo on message.mid=recipientinfo.mid group by message.mid;"
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
    query = "select m.mid, m.sender, group_concat(rvalue) as receivers, m.body from" + \
            "(select * from EnronAHS.message order by rand() limit " + str(num_emails) + \
            ")  as m inner join EnronAHS.recipientinfo on m.mid=recipientinfo.mid group by m.mid;"
    cur = db_conn.cursor()
    cur.execute("SET SESSION group_concat_max_len = 100000;")
    cur.execute(query)
    return cur.fetchall()