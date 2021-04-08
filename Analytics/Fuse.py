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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from functools import partial
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.svm import SVC
from scipy.stats import gaussian_kde
# from sklearn.metrics import det_curve
from Analytics import CMC_Plot as CMC
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')


class FuseRule:
    def __init__(self, list_o_rules, score_data, modalities, fusion_settings=None, experiment=''):
        self.list_o_rules = sorted(list(set(list_o_rules)))
        self.score_data = score_data
        self.modalities = modalities
        self.fused_modalities = []
        self.fusion_settings = fusion_settings
        self.results = pd.DataFrame(columns=['FPRS', 'TPRS', 'EER', 'AUC', 'thresholds'])

        self.experiment = experiment
        self.title = ''

    def make_rocs(self):
        if not os.path.exists('./generated/experiments/'):
            os.makedirs('./generated/experiments/')

        experiment_dir = self.experiment
        if not os.path.exists('./generated/experiments/' + experiment_dir):
            os.makedirs('./generated/experiments/' + experiment_dir)

        print('Making ROCs... ')
        plt.figure()

        analytic_beans = []

        ## Baseline
        for modality in self.modalities:
            metrics = {'FPRS': [], 'TPRS': [], 'EER': [], 'AUC': [], 'thresholds': []}
            Y_test = self.score_data['Label']
            scores = self.score_data[modality]

            fprs, tprs, thresh = roc_curve(Y_test, scores)

            metrics['FPRS'] = fprs
            metrics['TPRS'] = tprs
            metrics['thresholds'] = thresh
            metrics['EER'] = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
            metrics['AUC'] = roc_auc_score(Y_test, scores)
            analytic_beans.append(metrics)
            # plt.semilogx(fprs, tprs, label=modality, marker='+')

        ## Fused
        for modality in self.fused_modalities:
            metrics = {'FPRS': [], 'TPRS': [], 'EER': [], 'AUC': [], 'thresholds': []}
            Y_test = self.score_data['Label']
            scores = self.score_data[modality]

            fprs, tprs, thresh = roc_curve(Y_test, scores)

            metrics['FPRS'] = fprs
            metrics['TPRS'] = tprs
            metrics['thresholds'] = thresh
            metrics['EER'] = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
            metrics['AUC'] = roc_auc_score(Y_test, scores)
            analytic_beans.append(metrics)
            # plt.semilogx(fprs, tprs, label=modality, marker='X')

        self.results = pd.DataFrame(analytic_beans)
        self.results.index = [x for x in self.modalities] + [x for x in self.fused_modalities]

        ############## Baseline Plots
        for baseline in self.modalities:
            fprs = self.results.loc[baseline]['FPRS']
            tprs = self.results.loc[baseline]['TPRS']
            plt.semilogx(fprs, tprs, label=baseline, marker='+')


        plt.legend(bbox_to_anchor=(0.5, -0.02), loc="lower left", borderaxespad=0)
        plt.xlabel('False Match Rate (FMR)',  fontsize=15)
        plt.ylabel('True Match Rate (TMR)', fontsize=15)
        plt.title('Baseline Modalities', fontsize=15)

        plot_name = './generated/experiments/' + experiment_dir + '/baseline.png'
        plt.savefig(plot_name, bbox_inches='tight', pad_inches=0.5)
        plt.clf()

        ############## Fused Plots
        for fused in self.fused_modalities:
            for baseline in self.modalities:
                fprs = self.results.loc[baseline]['FPRS']
                tprs = self.results.loc[baseline]['TPRS']
                plt.semilogx(fprs, tprs, label=baseline, marker='+')

            fused_fprs =  self.results.loc[fused]['FPRS']
            fused_tprs =  self.results.loc[fused]['TPRS']
            plt.semilogx(fused_fprs, fused_tprs, label=fused, marker='X')

            plt.legend(bbox_to_anchor=(0.5, -0.02), loc="lower left", borderaxespad=0)
            plt.xlabel('False Match Rate (FMR)',  fontsize=15)
            plt.ylabel('True Match Rate (TMR)', fontsize=15)

            plt.title(fused + 'Fusion', fontsize=15)

            plot_name = './generated/experiments/' + experiment_dir + '/' + fused.replace(':', '') + '.png'
            plt.savefig(plot_name, bbox_inches='tight', pad_inches=0.5)
            plt.clf()


        ############## ALL Plots
        for baseline in self.modalities:
            fprs = self.results.loc[baseline]['FPRS']
            tprs = self.results.loc[baseline]['TPRS']
            plt.semilogx(fprs, tprs, label=baseline, marker='+')

        for fused in self.fused_modalities:
            fused_fprs =  self.results.loc[fused]['FPRS']
            fused_tprs =  self.results.loc[fused]['TPRS']
            plt.semilogx(fused_fprs, fused_tprs, label=fused, marker='X')

            plt.legend(bbox_to_anchor=(0.5, -0.02), loc="lower left", borderaxespad=0)
            plt.xlabel('False Match Rate (FMR)',  fontsize=15)
            plt.ylabel('True Match Rate (TMR)', fontsize=15)

            fusion_rules = [x for x in self.score_data.columns if x.isupper()]
            plt.title('' + ' '.join(fusion_rules), fontsize=15)

        plot_name = './generated/experiments/' + experiment_dir + '/' + 'all.png'
        plt.savefig(plot_name, bbox_inches='tight', pad_inches=0.5)
        plt.clf()

        print('Finished ROC')


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

    def sequential_rule(self):
        train_y = self.score_data[list(self.score_data)[0]]['train_y']
        test_y = self.score_data[list(self.score_data)[0]]['test_y']

        baseline = self.fusion_settings['baseline']

        modals=list(self.score_data.keys())
        other_modalities = [x for x in modals if x != baseline and 'Rule' not in x and 'TRAIN' not in x]

        train_x = np.array(self.score_data[baseline]['train_x'])
        test_x = np.array(self.score_data[baseline]['test_x'])

        if self.fusion_settings['auto']:
            # data_C.loc[data_C['Label'] == 1.0]['Data']
            gen_scores = test_x[test_y == 1.0]
            imp_scores = test_x[test_y == 0.0]
            alpha, beta = self.extract_alpha_beta(gen_scores, imp_scores)

        else:
            alpha = self.fusion_settings['alpha']
            beta = self.fusion_settings['beta']


        for score_index in range(len(train_x)):
            if alpha <= train_x[score_index] < beta:
                others = [train_x[score_index]]
                for mod in other_modalities:
                    others.append(self.score_data[mod]['train_x'][score_index])

                train_x[score_index] = np.average(np.array(others))

        for score_index in range(len(test_x)):
            if alpha <= test_x[score_index] < beta:
                others = [test_x[score_index]]
                for mod in other_modalities:
                    others.append(self.score_data[mod]['train_x'][score_index])

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

        # print("sequential RULE TOOK: " + str(time.time()-start_time) + ' seconds')
        self.title = self.title + 'Auto__'+ baseline + '-' +  str(int(alpha*100)) + '-' + str(int(beta*100))
        self.return_modals['SequentialRule'] = {'train_x': train_x,
                                       'train_y': train_y,
                                       'test_x': test_x,
                                       'test_y': test_y}

    def sum_rule(self):
        sum_title = 'SIMPLE_SUM:'
        self.fused_modalities.append(sum_title)
        self.score_data.insert(len(self.modalities), sum_title, self.score_data[self.modalities].mean(axis=1))

    def svm_rule(self):
        svm_title = 'SVM:'
        self.fused_modalities.append(svm_title)

        train_indices = self.score_data.index[self.score_data['Train_Test'] == 'TRAIN']
        test_indices = self.score_data.index[self.score_data['Train_Test'] == 'TEST']

        train = self.score_data.iloc[train_indices]
        train_y = self.score_data.iloc[train_indices]['Label']
        train_x = train[self.modalities]

        test = self.score_data.iloc[test_indices]
        test_x = test[self.modalities]

        clf = SVC(gamma='auto', probability=True)
        clf.fit(train_x, train_y)

        ## Get the scores for the predicted test labels
        preds = clf.predict(test_x).astype(int)
        test_scores_all = clf.predict_proba(test_x)
        rows = [i for i in range(len(preds))]
        test_scores = pd.DataFrame(test_scores_all[rows, preds], index=test_indices)

        train_scores_all = clf.predict_proba(train_x)
        rows = [i for i in range(len(train_x.index))]
        train_scores = pd.DataFrame(train_scores_all[rows, train_y.astype(int)], index=train_indices)

        scores = pd.concat([train_scores, test_scores])
        self.score_data[svm_title] = scores


    ################################################################################################################
    ################################################################################################################

    def fuse_all(self):
        """

        :param list_o_rules:
        :param modal_infos:
        :return:
        """
        sorted_rules = sorted(list(set(self.list_o_rules)))

        if 'SequentialRule' in sorted_rules:
            self.sequential_rule()

        if 'SumRule' in sorted_rules:
            self.sum_rule()

        if 'SVMRule' in sorted_rules:
            self.svm_rule()

        self.make_rocs()

        self.modalities.extend(self.fused_modalities)
        return self.results

    def cmc(self):
        cmc = CMC.CMC(data=self.score_data, modalities=self.modalities, k=20)
        cmc.chop_and_sort()
        cmc.plots()
