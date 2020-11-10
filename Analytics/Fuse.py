"""
Created on 3/9/2020
By Melissa Dale
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from functools import partial
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.svm import SVC
from scipy.stats import gaussian_kde
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

class FuseRule:
    def __init__(self, list_o_rules, modal_infos, norm=None, fusion_settings=None, matrix_form=False):
        self.list_o_rules = sorted(list(set(list_o_rules)))
        self.modal_info = modal_infos
        self.return_modals = defaultdict(lambda: defaultdict(partial(np.ndarray, 0)))

        self.norm = norm
        self.fusion_settings = fusion_settings
        self.matrix_form = matrix_form

        self.title = ''

    def make_rocs(self):
        """

        :param data_lists: Lists of scores (each item of list is a list for a modality)
        :param labels: Labels for plots
        :return:
        """
        if not os.path.exists('./generated/ROC/'):
            os.makedirs('./generated/ROC/')

        result_dict = defaultdict(lambda: defaultdict(partial(np.ndarray, 0)))
        plotz = []
        rulz = []

        for base, items in self.modal_info.items():
            self.return_modals[base] = items

        print('Making ROCs... ')
        plt.figure()

        for key, item in self.return_modals.items():
            rulz.append(key+'_')
            Y_test = self.return_modals[key]['test_y']
            scores = self.return_modals[key]['test_x']

            fprs, tprs, _ = roc_curve(Y_test, scores)
            result_dict[key]['fprs'] = fprs
            result_dict[key]['tprs'] = tprs
            # https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
            result_dict[key]['EER'] = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
            result_dict[key]['AUC'] = roc_auc_score(Y_test, scores)

            tmp_scores = np.array(scores)
            tmp_scores[tmp_scores < 0.5] = 0
            tmp_scores[tmp_scores > 0.5] = 1

            result_dict[key]['F1'] = f1_score(Y_test, tmp_scores.astype(int), average='binary')
            plt.semilogx(fprs, tprs, label=key.replace('-TEST', ''), marker='+')

        plt.legend(bbox_to_anchor=(0.5, -0.02), loc="lower left", borderaxespad=0)
        plt.xlabel('False Match Rate (FMR)',  fontsize=15)
        plt.ylabel('True Match Rate (TMR)', fontsize=15)

        plt.title(self.norm + ' Normalization', fontsize=15)

        plot_name = './generated/ROC/' + self.title + '.png'
        plotz.append(plot_name)
        plt.savefig(plot_name, bbox_inches='tight')


        print('Finished ROC')
        plt.clf()

        return result_dict, plotz

    ################################################################################################################
    ################################################################################################################
    def extract_alpha_beta(self, gen, imp):
        # get bin width for gen and imp
        gen_Q1 = np.quantile(gen, 0.25)
        gen_Q3 = np.quantile(gen, 0.75)
        gen_IQR = gen_Q3 - gen_Q1
        cube = np.cbrt(len(gen))
        gen_width = 2*gen_IQR/cube

        imp_Q1 = np.quantile(gen, 0.25)
        imp_Q3 = np.quantile(gen, 0.75)
        imp_IQR = imp_Q3 - imp_Q1
        cube_imp = np.cbrt(len(imp))
        imp_width = 2*imp_IQR/cube_imp

        # Dummy X specifies the values for the x axis
        dummy_x = np.linspace(0.0, 1.0, 100)
        gen_kde = gaussian_kde(gen, bw_method=gen_width)
        y_gen = gen_kde.evaluate(dummy_x)

        imp_kde = gaussian_kde(imp, bw_method=imp_width)
        y_imp = imp_kde.evaluate(dummy_x)
        y = y_imp / y_gen

        # # # # # #
        potential_x = []
        potential_y = []

        # "Near 1" range
        lower, upper = 1.0, 1.0

        while len(potential_y) <= 10:
            for i in range(len(y)):
                lower = lower - 0.001
                upper = upper + 0.001

                if lower <= y[i] <= upper:
                    potential_x.append(dummy_x[i])
                    potential_y.append(y[i])

        return min(potential_x), max(potential_x)

    def serial_rule(self):
        start_time = time.time()
        train_y = self.modal_info[list(self.modal_info)[0]]['train_y']
        test_y = self.modal_info[list(self.modal_info)[0]]['test_y']

        baseline = self.fusion_settings['baseline']

        modals=list(self.modal_info.keys())
        other_modalities = [x for x in modals if x != baseline and 'Rule' not in x and 'TRAIN' not in x]

        train_x = np.array(self.modal_info[baseline]['train_x'])
        test_x = np.array(self.modal_info[baseline]['test_x'])

        if self.fusion_settings['auto']:
            # data_C.loc[data_C['Label'] == 1.0]['Data']
            gen_scores = test_x[test_y == 1.0]
            imp_scores = test_x[test_y == 0.0]
            alpha, beta = self.extract_alpha_beta(gen_scores, imp_scores)
            print("Alpha: " + str(alpha))
            print("Beta: " + str(beta))
        else:
            alpha = self.fusion_settings['alpha']
            beta = self.fusion_settings['beta']


        for score_index in range(len(train_x)):
            if alpha <= train_x[score_index] < beta:
                others = [train_x[score_index]]
                for mod in other_modalities:
                    others.append(self.modal_info[mod]['train_x'][score_index])

                train_x[score_index] = np.average(np.array(others))

        for score_index in range(len(test_x)):
            if alpha <= test_x[score_index] < beta:
                others = [test_x[score_index]]
                for mod in other_modalities:
                    others.append(self.modal_info[mod]['train_x'][score_index])

                test_x[score_index] = np.average(np.array(others))

        # else:
            # precentage_tracker=0
            # precentage_tracker_gen = 0
            # precentage_tracker_imp = 0
            #
            # for score_index in range(len(test_x)):
            #     if alpha <= test_x[score_index] < beta:
            #         precentage_tracker += 1
            #         if test_y[score_index] == 1.0:
            #             precentage_tracker_gen +=1
            #
            #         if test_y[score_index] == 0.0:
            #             precentage_tracker_imp +=1
            #
            #         others = [test_x[score_index]]
            #         for mod in other_modalities:
            #             others.append(self.modal_info[mod]['test_x'][score_index])
            #         test_x[score_index] = np.average(np.array(others))
            #
            # perc_changed = precentage_tracker/len(test_x)
            # print('!!!!!!!!!!!!!!!!!!!!!!!!')
            # print('PERCENTAGE CHANGED:' + str(perc_changed))
            # print('Imposter Fused: ' + str(precentage_tracker_imp/precentage_tracker))
            # print('Genuine Fused: ' + str(precentage_tracker_gen/precentage_tracker))

        print("SERIAL RULE TOOK: " + str(time.time()-start_time) + ' seconds')
        self.title = self.title + 'Auto__'+ baseline + '-' +  str(int(alpha*100)) + '-' + str(int(beta*100))
        self.return_modals['SerialRule'] = {'train_x': train_x,
                                       'train_y': train_y,
                                       'test_x': test_x,
                                       'test_y': test_y}

    def sum_rule(self):
        start_time = time.time()
        train_y = self.modal_info[list(self.modal_info)[0]]['train_y']
        test_y = self.modal_info[list(self.modal_info)[0]]['test_y']
        summed_train_x = None
        summed_test_x = None

        for key, scores in self.modal_info.items():
            if summed_train_x is None:
                summed_train_x = scores['train_x']
                summed_test_x = scores['test_x']
            else:
                summed_train_x = np.vstack((summed_train_x, scores['train_x']))
                summed_test_x = np.vstack((summed_test_x, scores['test_x']))

        self.title = self.title + '-Sum-'
        self.return_modals['SumRule'] = {'train_x': summed_train_x.sum(axis=0), 'train_y': train_y,
                                 'test_x': summed_test_x.sum(axis=0), 'test_y': test_y}
        print("SUM RULE TOOK: " + str(time.time()-start_time) + ' seconds')


    def svm_rule(self):
        train_y = self.modal_info[list(self.modal_info)[0]]['train_y']
        test_y = self.modal_info[list(self.modal_info)[0]]['test_y']
        train_x = None
        test_x = None

        for key, scores in self.modal_info.items():
            if 'Rule' in key:
                continue
            if train_x is None:
                train_x = scores['train_x']
                test_x = scores['test_x']
            else:
                train_x = np.vstack((train_x, scores['train_x']))
                test_x = np.vstack((test_x, scores['test_x']))

        clf = SVC(gamma='auto', probability=True)
        clf.fit(train_x.transpose(), train_y)
        y_score = clf.predict_proba(test_x.transpose())

        tmp = []
        for i in range(len(test_y)):
            tmp.append(y_score[i][int(test_y[i])])
        y_score = tmp

        self.title = self.title + '-SVM-'
        self.return_modals['SVMRule'] = {'train_x': train_x, 'train_y': train_y,
                                 'test_x': y_score, 'test_y': test_y}


    ################################################################################################################
    ################################################################################################################

    def fuse_all(self):
        """

        :param list_o_rules:
        :param modal_infos:
        :return:
        """
        sorted_rules = sorted(list(set(self.list_o_rules)))

        if 'SerialRule' in sorted_rules:
            self.serial_rule()

        if 'SumRule' in sorted_rules:
            self.sum_rule()

        if 'SVMRule' in sorted_rules:
            self.svm_rule()

        eval_mets, tmp_plts = self.make_rocs()

        return eval_mets, tmp_plts

