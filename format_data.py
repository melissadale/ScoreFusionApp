import glob
import os
import pandas as pd
from collections import defaultdict
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import Norms as nms
from kivy.uix.progressbar import ProgressBar

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

def split_gen_imp(scores, matrix_form=False):
    scores = scores.values
    genuine = []
    imposter = []

    # MATRIX Loading
    if matrix_form:
        for i in range(len(scores)-1):
            genuine.append(scores[i][i])

            for j in range(len(scores)-1):
                if j != i:
                    imposter.append(scores[i][j])

    # COLUMN LOADING
    else:
        for i in range(len(scores)):
            if scores[i][-1] == 1:
                genuine.append(scores[i][0:-1])
            elif scores[i][-1] == 0:
                imposter.append(scores[i][0:-1])

    groomed_data = np.array(genuine + imposter)
    lbl = np.array([1.0 for i in range(len(genuine))] + [0.0 for i in range(len(imposter))])

    return groomed_data, lbl


def reverse_gen_imp(dat, y):
    gen = dat[y == 1.0]
    imp = dat[y == 0.0]

    return gen, imp


def make_density_plots(gen, imp, label='Test', norm_type='None', modality='', exp='', pb=None, pb_increment=None):
    print('Plotting Density Estimates ... ')

    if not os.path.exists('./generated/density/' + label + '/PDF/'):
        os.makedirs('./generated/density/' + label + '/PDF/')
    if not os.path.exists('./generated/density/' + label + '/hist/'):
        os.makedirs('./generated/density/' + label + '/hist/')
    if not os.path.exists('./generated/density/' + label + '/overlap/'):
        os.makedirs('./generated/density/' + label + '/overlap/')

    #################################################################
    # Overlaid
    #################################################################
    sns.distplot(imp, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='Imposter', color='#C89A58')
    sns.distplot(gen, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='Genuine', color='#0DB14B')
    ax = plt.gca()
    ax2 = plt.twinx()
    sns.distplot(imp, hist=True, kde=False,
                 kde_kws={'shade': False, 'linewidth': 6},
                 label='Imposter', color='#FF1493', ax=ax2)
    sns.distplot(gen, hist=True, kde=False,
                 kde_kws={'shade': False, 'linewidth': 6},
                 label='Genuine', color='#7B68EE', ax=ax2)

    p = 'Density Estimates for '+modality+'\n' + label + '\n ' + str(len(gen)) + ' Subjects '

    ax2.legend(bbox_to_anchor=(1, 1), loc='upper center')
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    lims = ax.get_xlim()
    y_ticks = ax.get_yticks()
    ax.set_ylabel(r"Density Estimate")
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.set_ylabel(r"Sample Counts")
    plt.title(p)
    # ax.tick_params(axis="y", direction="in", pad=-22)

    plt.savefig('./generated/density/' + label + '/overlap/' + norm_type + '_' + exp + '_' + label + '-' + modality + '.png', bbox_inches='tight')
    plt.clf()

    #################################################################
    # pdf
    #################################################################
    sns.distplot(imp, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='Imposter', color='#C89A58')
    sns.distplot(gen, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='Genuine', color='#0DB14B')

    p = 'Density Estimates for '+modality+'\n' + label + '\n ' + str(len(gen)) + ' Subjects '
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.ylabel(r"Density Estimate")
    ax = plt.gca()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.tick_params(axis="y", direction="in", pad=-22)

    plt.title(p)
    ax = plt.gca()
    ax.set_xlim(lims)
    plt.savefig('./generated/density/' + label + '/PDF/' + norm_type + '_' + exp + '_' + label + '-' + modality + '.png', bbox_inches='tight')
    plt.clf()

    #################################################################
    # histogram
    #################################################################
    sns.distplot(imp, hist=True, kde=False,
                 kde_kws={'shade': False, 'linewidth': 6},
                 label='Imposter', color='#FF1493')
    sns.distplot(gen, hist=True, kde=False,
                 kde_kws={'shade': False, 'linewidth': 6},
                 label='Genuine', color='#7B68EE')

    ax = plt.gca()
    ax2 = plt.twinx()
    sns.distplot(imp, hist=True, kde=False,
                 kde_kws={'shade': False, 'linewidth': 6},
                 label='Imposter', color='#FF1493')
    sns.distplot(gen, hist=True, kde=False,
                 kde_kws={'shade': False, 'linewidth': 6},
                 label='Genuine', color='#7B68EE')

    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    p = 'Density Estimates for ' + modality+ '\n' + label + '\n ' + str(len(gen)) + ' Subjects '

    ax.set_xlim(lims)
    ax.set_yticks(y_ticks)

    plt.title(p)
    ax.set_ylabel(r"Density Estimate")
    ax2.set_ylabel(r"Sample Counts")

    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='y', colors='white')

    plt.savefig('./generated/density/' + label + '/hist/' + norm_type + '_' + exp + '_' + label + '-' + modality + '.png', bbox_inches='tight')
    plt.clf()

    print('should have updated by: ' + str(pb_increment))
    print(pb.value)
    pb.update_bar(pb_increment)

def get_data(pth, test_perc=0.2, normalize=None, norm_params=[], progress_bar=None):
    """
    Assumptions: Each file has the same number of samples
    """
    modalities = defaultdict(lambda: defaultdict(partial(np.ndarray, 0)))
    idx_train = None
    exp_id = pth.split('\\')[-1]

    if not os.path.exists('./generated/'):
        os.makedirs('./generated/')


    ###########################################################
    #########       Build Data Dictionaries
    ###########################################################
    for filename in glob.glob(str(pth) + '/*'):
        key = filename.split('\\')[-1].split('.')[0]
        modality_keys = []

        train_loaded, test_loaded = False, False

        # handle csv, or txt files
        if filename.endswith('.txt'):
            data = pd.read_csv(filename, sep='\t', header=None)
        elif filename.endswith('.csv'):
            data = pd.read_csv(filename, index_col=0)

        # determine if data is a matrix with gen across the diagonal, or column with labels based
        if data.shape[0] == data.shape[1]:
            matrix_form = True
            data = data.drop(data.columns[-1], axis=1)
        else:
            matrix_form = False
            modality_keys = list(data.columns)

        data_x, data_y = split_gen_imp(data, matrix_form)

        ######### PULL DATA
        if 'train' in filename.lower():
            train_x = data_x
            train_y = data_y
            train_loaded = True

            key = key.replace('-TRAIN', '')

        elif 'test' in filename.lower():
            test_x = data_x
            test_y = data_y
            test_loaded = True

            key = key.replace('-TEST', '')

        else:
            train_loaded, test_loaded = True, True
            if idx_train is None:
                indices = np.arange(len(data_y))
                _, _, _, _, idx_train, idx_test = train_test_split(
                    data_x, data_y, indices, stratify=data_y, test_size=test_perc*.01)

            train_x = data_x[idx_train]
            train_y = data_y[idx_train]
            test_x = data_x[idx_test]
            test_y = data_y[idx_test]

        mods = []
        for m in modality_keys:
            if m.lower() != 'label' and m.lower() != 'class':
                # mods.append(key + '___' + m)
                mods.append(m)


        if train_loaded:
            if len(modality_keys) == 0:
                modalities[key]['train_x'] = train_x
                modalities[key]['train_y'] = train_y

            else:
                for k in range(len(mods)):
                    modalities[mods[k]]['train_x'] = train_x[:, k]
                    modalities[mods[k]]['train_y'] = train_y



        if test_loaded:
            if len(modality_keys) == 0:
                modalities[key]['test_x'] = test_x
                modalities[key]['test_y'] = test_y
            else:
                for k in range(len(mods)):
                    modalities[mods[k]]['test_x'] = test_x[:, k]
                    modalities[mods[k]]['test_y'] = test_y


    ##########
    ##  ok, now normalize and plot density plots
    #########

    # Calculate progress bar things
    tasks = len(modalities.keys())*3
    pb_incremental = 100/tasks

    for key, items in modalities.items():
        train_x = items['train_x']
        train_y = items['train_y']
        test_x = items['test_x']
        test_y = items['test_y']

        if normalize == 'MinMax':
            transformed_train_x, transformed_test_x = nms.my_minmax(train_x, test_x)

        elif normalize == 'ZScore':
            transformed_train_x, transformed_test_x = nms.my_zscore(train_x, test_x)

        elif normalize == 'Decimal':
            transformed_train_x, transformed_test_x = nms.my_decimal(train_x, test_x)

        elif normalize == 'Median':
            transformed_train_x, transformed_test_x = nms.my_med_mad(train_x, test_x)

        # elif normalize=='DSigmoid':
        #     train_x, test_x = nms.my_double_sigmoid(train_x, test_x)

        elif normalize == 'BiweightEstimator':
            transformed_train_x, transformed_test_x = nms.my_biweight(train_x, test_x)

        elif normalize == 'TanhEstimator':
            transformed_train_x, transformed_test_x = nms.my_tanh(train_x, test_x, norm_params[0], norm_params[1], norm_params[2])

        else:
            transformed_train_x = train_x
            transformed_test_x = test_x

        modalities[key]['train_x'] = transformed_train_x
        modalities[key]['train_y'] = train_y
        modalities[key]['test_x'] = transformed_test_x
        modalities[key]['test_y'] = test_y

        gen, imp = reverse_gen_imp(transformed_train_x, train_y)
        make_density_plots(gen, imp, label='Training', norm_type=normalize, modality=key, exp=exp_id, pb=progress_bar, pb_increment=pb_incremental)

        gen, imp = reverse_gen_imp(transformed_test_x, test_y)
        make_density_plots(gen, imp, label='Testing', norm_type=normalize, modality=key, exp=exp_id, pb = progress_bar, pb_increment=pb_incremental)

        gen, imp = reverse_gen_imp(np.append(transformed_train_x, transformed_test_x), np.append(train_y, test_y))
        make_density_plots(gen, imp, label='Entire', norm_type=normalize, modality=key, exp=exp_id, pb = progress_bar, pb_increment=pb_incremental)
    print('returning modalities')
    return modalities, exp_id, matrix_form

