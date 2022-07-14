import pandas as pd
from pandas.errors import MergeError
import numpy as np
import math
import statsmodels.api as sm
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns


class ScoreData():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = pd.DataFrame(columns=['PROBE_ID', 'GALLERY_ID', 'LABEL', 'TRAIN_TEST'])

    def load_data(self, path, training_split=None):
        split = False
        for file in glob.glob(path+'/*'):
            ext = file.split('.')[-1]
            if ext != 'csv' and ext != 'txt':
                continue
            else:
                df = pd.read_csv(file, index_col=0)

            if 'train' in file.lower():
                split = 'TRAIN'

            elif 'test' in file.lower():
                split = 'TEST'


            # TODO check for cases where there are not identities included
            # if isinstance(df.iloc[1,0], str):
                # df.index = df.iloc[0]
                # df = df.iloc[1:, 1:]

            # check if file is in column format or matrix format
            if df.shape[0] == df.shape[1]:
                modality = file.split('\\')[-1].replace('.csv', '').replace('.txt', '')

                tmp_scores = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
                tmp_scores = tmp_scores.stack().reset_index()
                tmp_scores.columns = ['PROBE_ID', 'GALLERY_ID', modality]

                # Set "genuine" and "imposter" labels
                tmp_scores['LABEL'] = 0.0
                tmp_scores.loc[tmp_scores.PROBE_ID == tmp_scores.GALLERY_ID, 'LABEL'] = 1.0
                tmp_scores.index = tmp_scores['PROBE_ID'] + '_' + tmp_scores['GALLERY_ID']

            # Otherwise, data is in column format
            else:
                # This is a custom Large Data Format (provided from a specific data provider)
                if all(x in df.columns for x in ['probe_file_id', 'probe_subject_id',
                                                   'candidate_file_id', 'candidate_subject_id', 'genuine_flag']):

                    tmp_scores = pd.DataFrame()
                    tmp_scores['PROBE_ID'] = df['probe_subject_id']
                    tmp_scores['Probe_File'] = df['probe_file_id']
                    tmp_scores['GALLERY_ID'] = df['candidate_subject_id']
                    tmp_scores['Gallery_File'] = df['candidate_file_id']
                    tmp_scores['LABEL'] = df['genuine_flag']

                else:  # TODO: Make generic column format handling
                    pass

            try:
                self.data = self.data.merge(tmp_scores, how='outer')
            except MergeError:
                self.data = tmp_scores
                self.data['TRAIN_TEST'] = None

            if split:
                self.data['TRAIN_TEST'] = split

        if training_split:
            # ToDo change this so that it's subject disjoint (without using train_test_split method
            test_perc = (100 - training_split) / 100

            indices = np.arange(len(self.data.index))
            _, _, _, _, idx_train, idx_test = train_test_split(
                self.data[self.get_modalities()], self.data['LABEL'], indices, stratify=self.data['LABEL'],
                test_size=test_perc)

            self.data.iloc[idx_train, self.data.columns.get_loc("TRAIN_TEST")] = 'TRAIN'
            self.data.iloc[idx_test, self.data.columns.get_loc("TRAIN_TEST")] = 'TEST'

            # self.data['TRAIN_TEST'].iloc[idx_train] = 'TRAIN'
            # self.data['TRAIN_TEST'].iloc[idx_test] = 'TEST'

    def normalize_scores(self, norm_type, norm_params=None):
        for mod in self.get_modalities():
            if norm_type == 'MinMax':
                min_x = self.data[self.data['TRAIN_TEST'] == 'TRAIN'][mod].min()
                max_x = self.data[self.data['TRAIN_TEST'] == 'TRAIN'][mod].max()

                # trim test data to training's min and max values.
                self.data.loc[(self.data['TRAIN_TEST'] == 'TEST') &
                                    (self.data[mod] > max_x), mod] = max_x
                self.data.loc[(self.data['TRAIN_TEST'] == 'TEST') &
                                    (self.data[mod] < min_x), mod] = min_x

                self.data[mod] = (self.data[mod] - min_x) / (max_x - min_x)

            elif norm_type == 'ZScore':
                u = self.data[self.data['TRAIN_TEST'] == 'TRAIN'][mod].mean()
                s = self.data[self.data['TRAIN_TEST'] == 'TRAIN'][mod].std()
                self.data[mod] = (self.data[mod] - u) / (s)

            elif norm_type == 'Decimal':
                max_x = self.data[self.data['TRAIN_TEST'] == 'TRAIN'][mod].max()
                n = math.log10(max_x)
                self.data[mod] = (self.data[mod]) / (10 ** n)


            elif norm_type == 'Median':
                mad = self.data[self.data['TRAIN_TEST'] == 'TRAIN'][mod].mad()
                med = self.data[self.data['TRAIN_TEST'] == 'TRAIN'][mod].median()
                self.data[mod] = (self.data[mod] - med) / (mad)

            elif norm_type == 'DSigmoid':
                r1 = norm_params['r1']
                r2 = norm_params['r2']
                t = norm_params['t']


                exp = self.data[mod] - t
                r1_exp = np.exp(-2 * (exp / r1))
                r2_exp = np.exp(-2 * (exp / r2))
                self.data.loc[self.data[mod] < t, mod] = 1 / (1 + r1_exp)
                self.data.loc[self.data[mod] >= t, mod] = 1 / (1 + r2_exp)

            elif norm_type == 'BiweightEstimator':  # TODO
                pass

            elif norm_type == 'TanhEstimator':
                c = norm_params

                psi = sm.robust.norms.TukeyBiweight(c=c).psi(self.data[
                                                                 (self.data['TRAIN_TEST'] == 'TRAIN') &
                                                                 (self.data['LABEL'] == 1.0)][mod])
                MUgh = psi.mean()
                SIGgh = psi.std()

                self.data[mod] = 0.5 * (np.tan(0.01 * ((self.data[mod] - MUgh) / (SIGgh + 0.0000000001)) + 1))

            elif norm_type == 'None':
                pass

            else:
                print("ERROR: unrecognized normalization technique requested")

    def get_modalities(self):
        not_mods = ['PROBE_ID', 'GALLERY_ID', 'LABEL', 'TRAIN_TEST']
        return [x for x in self.data.columns if x not in not_mods and ':' not in x]

    def impute(self, impute_method):
        if not impute_method:
            # Listwise Deletion
            self.data = self.data[self.data[self.get_modalities()].notna()]
            return

        if type(impute_method) == list:
            k = impute_method[1]
            impute_method = impute_method[0]

        imp = None
        if impute_method == 'Mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')

        elif impute_method == 'Median':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')

        elif impute_method == 'Bayesian':
            imp = IterativeImputer(random_state=0, estimator=BayesianRidge())

        elif impute_method == 'DT':
            imp = IterativeImputer(random_state=0, estimator=DecisionTreeRegressor())

        elif impute_method == 'KNN':
            imp = IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=k))

        imp.fit(self.data[self.data['TRAIN_TEST'] == 'TRAIN'][self.get_modalities()])
        self.data[self.get_modalities()] = imp.transform(self.data[self.get_modalities()])

    def make_density_plots(self, modality):
        if not os.path.exists('./generated/density/PDF/'):
            os.makedirs('./generated/density/PDF/')
        if not os.path.exists('./generated/density/hist/'):
            os.makedirs('./generated/density/hist/')
        if not os.path.exists('./generated/density/overlap/'):
            os.makedirs('./generated/density/overlap/')

        gen = self.data[self.data['LABEL'] == 1.0][modality]
        imp = self.data[self.data['LABEL'] == 0.0][modality]
        #################################################################
        # Overlaid
        #################################################################
        sns.kdeplot(imp, fill=True, label='Imposter', color='#C89A58')
        sns.kdeplot(gen, fill=True, label='Genuine', color='#0DB14B')
        ax = plt.gca()
        ax2 = plt.twinx()
        sns.histplot(imp, kde=False, label='Imposter', color='#FF1493')
        sns.histplot(gen, kde=False, label='Genuine', color='#7B68EE')

        p = 'Density Estimates and Score Counts for ' + modality + '\n' + str(len(gen)) + ' Subjects '

        ax2.legend(bbox_to_anchor=(1, 1), loc='upper center')
        plt.legend(bbox_to_anchor=(1, 1), loc=2)
        lims = ax.get_xlim()
        y_ticks = ax.get_yticks()
        ax.set_ylabel(r"Density Estimate")
        ax2.set_ylabel(r"Sample Counts")

        ax.set_xlabel(modality)
        plt.title(p)

        plt.savefig('./generated/density/overlap/' + modality + '.png', bbox_inches='tight')
        plt.clf()

        #################################################################
        # pdf
        #################################################################
        sns.kdeplot(imp, fill=True, label='Imposter', color='#C89A58')
        sns.kdeplot(gen, fill=True, label='Genuine', color='#0DB14B')

        p = 'Density Estimates for ' + modality + '\n ' + str(len(gen)) + ' Subjects '
        plt.legend(bbox_to_anchor=(1, 1), loc=2)
        plt.ylabel(r"Density Estimate")
        plt.title(p)
        ax = plt.gca()
        ax.set_xlim(lims)
        ax.set_xlabel(modality)
        plt.savefig(
            './generated/density/PDF/' + modality + '.png',
            bbox_inches='tight')
        plt.clf()

        #################################################################
        # histogram
        #################################################################

        sns.histplot(imp, kde=False, label='Imposter', color='#FF1493')
        sns.histplot(gen, kde=False, label='Genuine', color='#7B68EE')
        ax = plt.gca()

        plt.legend(bbox_to_anchor=(1, 1), loc=2)
        p = 'Score counts for ' + modality + '\n' + str(len(gen)) + ' Subjects '

        ax.set_xlim(lims)
        ax.set_yticks(y_ticks)

        plt.title(p)
        ax.set_ylabel(r"Density Estimate")
        ax2.set_ylabel(r"Sample Counts")

        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='y', colors='white')

        ax.set_xlabel(modality)

        plt.savefig('./generated/density/hist/' + modality + '.png', bbox_inches='tight')
        plt.clf()
