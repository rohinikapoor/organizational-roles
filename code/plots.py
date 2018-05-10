import matplotlib as mpl;

mpl.use('Agg')  # The cluster cannot process other graphic engines
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import seaborn as sb
import time

import constants
import metrics_utils

from datetime import datetime
from sklearn.manifold import TSNE


def _assign_labels_colors(labels, colors):
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
            col = colors[col_idx % len(colors)]
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
    print('time taken by TSNE ', (end - start))

    # creating the colors
    colors = list(sb.color_palette().as_hex())
    color_list = _assign_labels_colors(labels, colors)

    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_embs[:, 0], tsne_embs[:, 1], c=color_list, s=75)
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


def plot_error_distribution(sender, train_errors, val_errors, test_errors):
    plt.close()
    x = [train_errors, val_errors, test_errors]
    plt.hist(x, 10, histtype='bar', color=['b', 'g', 'r'])

    plt.title(sender)
    plt.savefig('../outputs/{}-l2-errors-{}.png'.format(constants.RUN_ID, sender))


def plot_error_distribution_v2(model, w2v, sender, train, val, test):
    _, train_errors = metrics_utils.get_predictions(model, w2v, train, neg_emails=[], is_l2=True)
    _, val_errors = metrics_utils.get_predictions(model, w2v, val, neg_emails=[], is_l2=True)
    _, test_errors = metrics_utils.get_predictions(model, w2v, test, neg_emails=[], is_l2=True)
    plot_error_distribution(sender, train_errors, val_errors, test_errors)


# The following code was used to generate charts for the poster
def plot_bar_charts(labels, vals, ylabel, title, ymax, baseline=None, display_plot=False):
    # objects = ('CEO/Presidents', 'Employees', 'Directors', 'Managers', 'Others')
    # objects = ('-1', '0', '+1')
    plt.rc('ytick', labelsize=15)
    x_pos = np.arange(len(labels))
    # acc = [60.47, 48.60, 43.47]
    if baseline is not None:
        line_x = np.arange(-1, len(labels) + 1)
        baseline_y = np.zeros(len(line_x)) + baseline
        line, = plt.plot(line_x, baseline_y, color='k', label='Baseline', linewidth=2)
        plt.legend()
    plt.bar(x_pos, vals, align='center', width=0.3, color=['#EC7063', '#2ECC71', '#3498DB'])  # color='#3498DB')
    plt.xticks(x_pos, labels)
    plt.ylabel(ylabel)
    plt.ylim((0, ymax))
    # plt.title('Dominance')
    plt.title(title)
    plt.tight_layout()

    if display_plot:
        plt.show()
    else:
        outfile = '../outputs/' + title + '.png'
        plt.savefig(outfile)


def plot_bar_charts_v2(labels, vals, ylabel, title, ymax, display_plot=False):
    cols = vals.shape[0]
    N = 2 * vals.shape[1] + 1

    colours = ['#EC7063', '#2ECC71', '#3498DB']
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots()

    legend_colours = []
    for col in xrange(cols):
        this_vals = np.zeros(N)
        this_vals[1:N:2] = vals[col, :]
        this_rects = ax.bar(ind + col * width, this_vals, width, color=colours[col % len(colours)])
        legend_colours.append(this_rects[0])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('', 'Positive', '', 'Negative', ''))

    ax.legend(legend_colours, labels)

    plt.ylim((0, ymax))
    plt.tight_layout()

    if display_plot:
        plt.show()
    else:
        outfile = '../outputs/' + title + '.png'
        plt.savefig(outfile)


if __name__ == '__main__':
    labels = ['SR Model', 'PV Model', 'Discriminative']
    ymax = 1.0

    title = 'Hits @ 1000'
    vals = np.array([[0.57, 0.38],
                     [0.74, 0.50],
                     [0.529, 0.514]])
    ylabel = 'Hit ratio'
    plot_bar_charts_v2(labels, vals, ylabel, title, ymax, display_plot=False)

    # labels = ['SR Model', 'PV Model', 'Discriminative']
    # vals = [60.47, 48.60, 54.94]
    # ylabel = 'Accuracy'
    # title = 'Accuracy for Hierarchy Relations'
    # plot_bar_charts(labels, vals, ylabel, title, ymax=100.0, baseline=37.64)
    #
    # title = 'AP @ 1000'
    # vals = np.array([[0.30, 0.16],
    #                  [0.58, 0.28],
    #                  [0.29, 0.28]])
    # ylabel = 'Average Precision'
    # plot_bar_charts_v2(labels, vals, ylabel, title, ymax, display_plot=False)
    #
    # labels = ['SR Model', 'PV Model', 'Discriminative']
    # vals = [60.47, 48.60, 54.94]
    # ylabel = 'Accuracy'
    # title = 'Accuracy for Hierarchy Relations'
    # plot_bar_charts(labels, vals, ylabel, title, ymax=100.0, baseline=37.64)
    #
    # labels = ['Lower', 'Equal', 'Higher']
    # vals = [36.12, 26.23, 37.64]  # 4186
    # ylabel = 'Percentage Distribution'
    # title = 'Class Distribution for Hierarchy Relation'
    # plot_bar_charts(labels, vals, ylabel, title, ymax=50, baseline=None)
    #
    # labels = ['SR Model', 'PV Model', 'Discriminative']
    # vals = [48.69, 43.47, 42.6]
    # ylabel = 'Accuracy'
    # title = 'Accuracy for Role Prediction'
    # plot_bar_charts(labels, vals, ylabel, title, ymax=100.0, baseline=34.78)
    #
    # labels = ['CEO/Presidents', 'Directors', 'Employee', 'Manager', 'Others']
    # vals = [26.95, 13.91, 34.78, 12.17, 12.17]  # [31, 16, 40, 14, 14]
    # ylabel = 'Percentage Distribution'
    # title = 'Class Distribution for Role Prediction'
    # plot_bar_charts(labels, vals, ylabel, title, ymax=50, baseline=None)
    #
    # labels = ['CEO/Presidents', 'Directors', 'Employee', 'Manager', 'Others']
    # vals = [0.67, 0.0, 0.56, 0.0, 0.0]
    # ylabel = 'F-score'
    # title = 'Class-wise F-score for Role Prediction'
    # plot_bar_charts(labels, vals, ylabel, title, ymax=1.0, baseline=None)
    #
    # labels = ['Lower', 'Equal', 'Higher']
    # vals = [0.72, 0.20, 0.71]
    # ylabel = 'F-score'
    # title = 'Class-wise F-score for Hierarchy Relation'
    # plot_bar_charts(labels, vals, ylabel, title, ymax=1.0, baseline=None)
    #
    # labels = ['SR Model', 'PV Model', 'Discriminative']
    # ymax = 0.8
    #
    # title = 'Hits @ 1000'
    # vals = np.array([[0.57, 0.38],
    #                  [0.74, 0.50],
    #                  [0.53, 0.51]])
    # ylabel = 'Hit ratio'
    # plot_bar_charts_v2(labels, vals, ylabel, title, ymax, display_plot=False)
    #
    # title = 'AP @ 1000'
    # vals = np.array([[0.30, 0.16],
    #                  [0.58, 0.28],
    #                  [0.29, 0.28]])
    # ylabel = 'Average Precision'
    # plot_bar_charts_v2(labels, vals, ylabel, title, ymax, display_plot=False)

    # email_ids, embs = load_user_embeddings('../important_embeddings/embeddings_usr50d_em100d_custom_25ep_m3/embeddings_usr50d_em100d_custom_25ep_m3.pkl')
    # embs, labels = extract_emb_desgn(email_ids, embs, cat='cat1')
    # plot_with_tsne(labels.tolist(), embs)
