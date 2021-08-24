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
from Analytics import IdentificationTasks as Identify
import warnings
import time
from Analytics.Experiment import TrainedModel

warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')


class FuseRule:
    def __init__(self, list_o_rules, score_data, modalities, fusion_settings=None, experiment='', tasks=[], thinking=None):
        self.list_o_rules = sorted(list(set(list_o_rules)))
        self.score_data = score_data
        self.modalities = modalities
        self.fused_modalities = []
        self.fusion_settings = fusion_settings
        self.results = None
        self.tasks = tasks

        self.experiment_name = experiment
        self.title = ''
        self.models = []
        self.thinking = thinking

        self.cmc_accuracies = None

    def make_rocs(self):
        experiment_dir = self.experiment_name
        if not os.path.exists('./generated/experiments/ROC/' + experiment_dir):
            os.makedirs('./generated/experiments/ROC/' + experiment_dir)

        # TODO: handle ROC/CMC settings better
        if not os.path.exists('./generated/experiments/CMC/' + experiment_dir):
            os.makedirs('./generated/experiments/CMC/' + experiment_dir)


        print('Making ROCs... ')
        plt.figure()

        analytic_beans = []

        base_modals = [x for x in self.modalities if ':' not in x]
        fused_modals = [x for x in self.fused_modalities]


        ## Baseline
        if not self.results:
            for modality in base_modals:
                metrics = {'FPRS': [], 'TPRS': [], 'EER': [], 'AUC': [], 'thresholds': [], 'Experiment': []}
                Y_test = self.score_data['Label']
                scores = self.score_data[modality]

                fprs, tprs, thresh = roc_curve(Y_test, scores)

                metrics['Experiment'] = 'BASELINE'
                metrics['FPRS'] = fprs
                metrics['TPRS'] = tprs
                metrics['thresholds'] = thresh
                metrics['EER'] = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
                metrics['AUC'] = roc_auc_score(Y_test, scores)
                analytic_beans.append(metrics)

        ## Fused
        for modality in fused_modals:
            metrics = {'FPRS': [], 'TPRS': [], 'EER': [], 'AUC': [], 'thresholds': [], 'Experiment': []}
            Y_test = self.score_data['Label']
            scores = self.score_data[modality]

            fprs, tprs, thresh = roc_curve(Y_test, scores)

            metrics['Experiment'] = self.experiment_name
            metrics['FPRS'] = fprs
            metrics['TPRS'] = tprs
            metrics['thresholds'] = thresh
            metrics['EER'] = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
            metrics['AUC'] = roc_auc_score(Y_test, scores)
            analytic_beans.append(metrics)

        self.results = pd.DataFrame(analytic_beans)
        self.results.index = [x for x in base_modals] + [x for x in fused_modals]

        ############## Baseline Plots
        for baseline in base_modals:
            fprs = self.results.loc[baseline]['FPRS']
            tprs = self.results.loc[baseline]['TPRS']

            plt.semilogx(fprs, tprs, label=baseline, marker='+')


        plt.legend(bbox_to_anchor=(0.5, -0.02), loc="lower left", borderaxespad=0)
        plt.xlabel('False Match Rate (FMR)',  fontsize=15)
        plt.ylabel('True Match Rate (TMR)', fontsize=15)
        plt.title('Baseline Modalities', fontsize=15)

        plot_name = './generated/experiments/ROC/' + experiment_dir + '/baseline.png'
        plt.savefig(plot_name, bbox_inches='tight', pad_inches=0.5)
        plt.clf()

        ############## Fused Plots

        for fused in fused_modals:
            for baseline in base_modals:
                fprs = self.results.loc[baseline]['FPRS']
                tprs = self.results.loc[baseline]['TPRS']

                plt.semilogx(fprs, tprs, label=baseline, marker='+')

            fused_fprs =  self.results.loc[fused]['FPRS']
            fused_tprs =  self.results.loc[fused]['TPRS']
            plt.semilogx(fused_fprs, fused_tprs, label=fused, marker='X')

            plt.legend(bbox_to_anchor=(0.5, -0.02), loc="lower left", borderaxespad=0)
            plt.xlabel('False Match Rate (FMR)',  fontsize=15)
            plt.ylabel('True Match Rate (TMR)', fontsize=15)

            plt.title(fused + ' Fusion', fontsize=15)

            plot_name = './generated/experiments/ROC/' + experiment_dir + '/' + fused.replace(':', '') + '.png'
            plt.savefig(plot_name, bbox_inches='tight', pad_inches=0.5)
            plt.clf()


        ############## ALL Plots
        for baseline in base_modals:
            fprs = self.results.loc[baseline]['FPRS']
            tprs = self.results.loc[baseline]['TPRS']
            plt.semilogx(fprs, tprs, label=baseline, marker='+')

        for fused in fused_modals:
            fused_fprs =  self.results.loc[fused]['FPRS']
            fused_tprs =  self.results.loc[fused]['TPRS']
            plt.semilogx(fused_fprs, fused_tprs, label=fused, marker='X')

            plt.legend(bbox_to_anchor=(0.5, -0.02), loc="lower left", borderaxespad=0)
            plt.xlabel('False Match Rate (FMR)',  fontsize=15)
            plt.ylabel('True Match Rate (TMR)', fontsize=15)

            fusion_rules = [x.replace(':', '') for x in self.score_data.columns if x.isupper()]
            plt.title('All Fusion Rules::' + ' '.join(fusion_rules), fontsize=15)

        plot_name = './generated/experiments/ROC/' + experiment_dir + '/' + 'all.png'
        plt.savefig(plot_name, bbox_inches='tight', pad_inches=0.5)
        plt.clf()

        print('Finished ROC')


    ################################################################################################################
    ################################################################################################################
    def extract_alpha_beta(self, gen, imp):
        # get bin width for gen and imp
        gen_Q1 = np.quantile(gen, 0.25)
        imp_Q3 = np.quantile(imp, 0.75)

        ran = np.array([imp_Q3, gen_Q1])

        return min(ran), max(ran)

    def sequential_rule(self):
        t0 = time.time()
        seq_title='SequentialRule'
        train = self.score_data[self.score_data['Train_Test'] == 'TRAIN']

        baseline = self.fusion_settings['baseline']

        if self.fusion_settings['auto']:
            seq_title = seq_title+'(AUTO):'

            gen_scores = train[train['Label'] == 1.0][baseline]
            imp_scores = train[train['Label'] == 0.0][baseline]
            alpha, beta = self.extract_alpha_beta(gen_scores, imp_scores)

        else:
            alpha = self.fusion_settings['alpha']
            beta = self.fusion_settings['beta']
            seq_title = seq_title+'('+str(round(alpha, 2))+'-'+str(round(alpha, 2))+'):'

        if seq_title not in self.fused_modalities:
            self.fused_modalities.append(seq_title)


        self.score_data[seq_title] = \
            self.score_data[(self.score_data[baseline]>= alpha) & (self.score_data[baseline]< beta)][self.modalities].mean(axis=1)
        self.score_data[seq_title].fillna(self.score_data[baseline], inplace=True)

        t1 = time.time()
        self.models.append(TrainedModel(title=seq_title, train_time=t1 - t0, model=None))

    def sum_rule(self):
        t0 = time.time()
        sum_title = 'SIMPLE_SUM:'

        if sum_title not in self.fused_modalities:
            self.fused_modalities.append(sum_title)

        self.score_data[sum_title] = self.score_data[self.modalities].mean(axis=1)
        t1 = time.time()

        self.models.append(TrainedModel(title=sum_title, train_time=t1-t0, model=None))


    def svm_rule(self):
        t0 = time.time()
        svm_title = 'SVM:'

        if svm_title not in self.fused_modalities:
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
        t1 = time.time()

        self.models.append(TrainedModel(title=svm_title, train_time=t1-t0, model=clf))

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

        if 'Verification' in self.tasks:
            self.make_rocs()

        if 'Identication' in self.tasks:
            if not os.path.exists('./generated/experiments/CMC/' + self.experiment_name):
                os.makedirs('./generated/experiments/CMC/' + self.experiment_name)
            self.cmc_accuracies = self.cmc()

        # self.modalities.extend(self.fused_modalities)
        return self.results, self.models, self.cmc_accuracies

    def cmc(self):
        cmc = Identify.Identify(data=self.score_data, modalities=self.modalities, fused_modalities=self.fused_modalities, k=20, exp_id=self.experiment_name)
        cmc.chop_and_sort()
        cmc.generate_plots()
        # return cmc.get_accuracies()
        return cmc.get_cmc_summary()
