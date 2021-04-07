import glob
import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Score_data:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        self.test_perc = kwargs.get('test_perc')
        self.normalize = kwargs.get('normalize')
        self.norm_param = kwargs.get('norm_param')

        self.colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
        self.score_data = pd.DataFrame(columns=['Probe_Ids', 'Gallery_Ids', 'Train_Test'])
        self.score_data_inactive = pd.DataFrame()
        self.modalities = []
        self.lbl = kwargs.get('lbl')


    def make_density_plots(self, gen, imp, label='', norm_type='None', modality='', exp=''):
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
                     label='Imposter', color='#FF1493')
        sns.distplot(gen, hist=True, kde=False,
                     kde_kws={'shade': False, 'linewidth': 6},
                     label='Genuine', color='#7B68EE')

        p = 'Density Estimates and Score Counts for '+modality+'\n' + label + '\n ' + str(len(gen)) + ' Subjects '

        ax2.legend(bbox_to_anchor=(1, 1), loc='upper center')
        plt.legend(bbox_to_anchor=(1, 1), loc=2)
        lims = ax.get_xlim()
        y_ticks = ax.get_yticks()
        ax.set_ylabel(r"Density Estimate")
        ax2.set_ylabel(r"Sample Counts")
        plt.title(p)

        plt.savefig('./generated/density/' + label + '/overlap/' + norm_type + '_' + exp + '_' + label + '-' + modality
                    + '.png', bbox_inches='tight')
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
        p = 'Score counts for ' + modality+ '\n' + label + '\n ' + str(len(gen)) + ' Subjects '

        ax.set_xlim(lims)
        ax.set_yticks(y_ticks)

        plt.title(p)
        ax.set_ylabel(r"Density Estimate")
        ax2.set_ylabel(r"Sample Counts")

        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='y', colors='white')

        plt.savefig('./generated/density/' + label + '/hist/' + norm_type + '_' + exp + '_' + label + '-' + modality + '.png', bbox_inches='tight')
        plt.clf()

    def load_data(self):
        """
        Assumptions: Each file has the same number of samples
        """

        exp_id = self.path.split('\\')[-1]

        if not os.path.exists('./generated/'):
            os.makedirs('./generated/')

        ###########################################################
        #########       Build Data Frame
        ###########################################################
        file_data = pd.DataFrame(columns=['Probe_ID', 'Gallery_ID', 'Label', 'Train_Test'])
        for filename in glob.glob(str(self.path) + '/*'):
            key = filename.split('\\')[-1].split('.')[0]

            # handle csv, or txt files
            if filename.endswith('.txt'):
                data = pd.read_csv(filename, sep='\t', header=None)
            elif filename.endswith('.csv'):
                data = pd.read_csv(filename, index_col=0)
            else: # not a score file
                continue

            if 'train' in filename.lower():
                file_data['Train_Test'] = 'TRAIN'
                key = key.replace('-TRAIN', '')

            elif 'test' in filename.lower():
                file_data['Train_Test'] = 'TEST'
                key = key.replace('-TEST', '')

            if data.shape[0] == data.shape[1]: # TODO
                ## take triangle and flatten into row - column - value format
                tmp_scores = data.where(np.triu(np.ones(data.shape)).astype(np.bool))
                tmp_scores = tmp_scores.stack().reset_index()
                tmp_scores.columns = ['Probe_Ids', 'Gallery_ID', 'Value']

                gens = tmp_scores[tmp_scores['Probe_Ids'] == tmp_scores['Gallery_ID']]
                imps = tmp_scores[tmp_scores['Probe_Ids'] != tmp_scores['Gallery_ID']]

                # if there are alphanumeric values, assume these are id's
                # if is_string_dtype(gens['Row']) and is_string_dtype(gens['Column']):

                gen_datas = pd.DataFrame()
                gen_datas['Probe_ID'] = gens['Probe_Ids']
                gen_datas['Gallery_ID'] = gens['Gallery_ID']
                gen_datas['Label'] = 1.0
                gen_datas[key] = gens['Value']

                imp_datas = pd.DataFrame()
                imp_datas['Probe_ID'] = imps['Probe_Ids']
                imp_datas['Gallery_ID'] = imps['Gallery_ID']
                imp_datas['Label'] = 0.0
                imp_datas[key] = imps['Value']

                tmp_file_data = pd.concat([gen_datas, imp_datas], ignore_index=True)


            # Otherwise, data is in column format
            else:
                # This is MITRE data
                if all(x in data.columns for x in ['probe_file_id', 'probe_subject_id',
                                                   'candidate_file_id', 'candidate_subject_id', 'genuine_flag']):

                    tmp_file_data = pd.DataFrame()
                    tmp_file_data['Probe_Ids'] = data['probe_subject_id']
                    tmp_file_data['Probe_File'] = data['probe_file_id']
                    tmp_file_data['Gallery_Ids'] = data['candidate_subject_id']
                    tmp_file_data['Gallery_File'] = data['candidate_file_id']
                    tmp_file_data['Label'] = data['genuine_flag']
                    tmp_file_data[key] = data['score']

                else: # TODO
                    self.score_data[key] = data['score']

            file_data = file_data.merge(tmp_file_data, how='outer')
        modality_list = [x for x in file_data if x not in ['Label', 'Probe_ID', 'Gallery_ID', 'Train_Test'] and
                         '_ORIGINAL' not in x]

        # IF there are any NANs in the Train_Test column, redivide the data
        # TODO: ensure subject disjoint
        if file_data['Train_Test'].isnull().values.any():
            indices = np.arange(len(file_data.index))
            _, _, _, _, idx_train, idx_test = train_test_split(
                file_data[modality_list], file_data['Label'], indices, stratify=file_data['Label'],
                test_size=self.test_perc * .01)

            file_data['Train_Test'].iloc[idx_train] = 'TRAIN'
            file_data['Train_Test'].iloc[idx_test] = 'TEST'

        self.score_data = file_data
        self.modalities = modality_list
        return


    def normalize_data(self):
        for mod in self.score_data[self.modalities]:
            tmp = self.score_data[mod]
            self.score_data.rename(columns={mod: mod + '_ORIGINAL'}, inplace=True)

            self.score_data[mod] = tmp

            if self.normalize == 'MinMax':
                min_x = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][mod].min()
                max_x = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][mod].max()

                # trim test data to training's min and max values.
                self.score_data.loc[(self.score_data['Train_Test'] == 'TEST') &
                                    (self.score_data[mod] > max_x), mod] = max_x
                self.score_data.loc[(self.score_data['Train_Test'] == 'TEST') &
                                    (self.score_data[mod] < min_x), mod] = min_x

                self.score_data[mod] = (self.score_data[mod] - min_x) / (max_x - min_x)

            elif self.normalize == 'ZScore':
                u = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][mod].mean()
                s = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][mod].std()
                self.score_data[mod] = (self.score_data[mod] - u) / (s)

            elif self.normalize == 'Decimal':
                max_x = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][mod].max()
                n = math.log10(max_x)
                self.score_data[mod] = (self.score_data[mod]) / (10**n)


            elif self.normalize == 'Median':
                mad = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][mod].mad()
                med = self.score_data[self.score_data['Train_Test'] == 'TRAIN'][mod].median()
                self.score_data[mod] = (self.score_data[mod]-med) / (mad)

            elif self.normalize=='DSigmoid': # TODO
                # r1 = self.norm_params['r1']
                # r2 = self.norm_params['r2']
                # t = self.norm_params['t']
                #
                # self.score_data[self.score_data[key]< t] = 1 / (1 + np.exp(-2((self.score_data[key]-t) / r1)))
                # self.score_data[self.score_data[key] >= t] = 1 / (1 + np.exp(-2((self.score_data[key] - t) / r2)))
                pass

            elif self.normalize == 'BiweightEstimator':
                print("Biweight Estimator is not implemented yet")

            elif self.normalize == 'TanhEstimator':
                psi = sm.robust.norms.TukeyBiweight(c=4.685).psi(self.score_data[self.score_data['Train_Test']
                                                                                 == 'TRAIN'][mod])
                MUgh = psi.mean()
                SIGgh = psi.std()

                self.score_data[mod] = 0.5 * (np.tan(0.01 * ((self.score_data[mod] - MUgh) / SIGgh)) + 1)

            else:
                print("ERROR: unrecognized normalization technique requested")


    def plot_distributions(self): ## ToDO move to GUI.py
        # DIST PLOTS
        exp_id = ''
        for mod in self.modalities:
            # key = 'NORMALIZED_' + mod
            # Clock.schedule_once(partial(self.update_bar, ''))

            # Training Data
            gens = self.score_data.loc[(self.score_data['Train_Test'] == 'TRAIN') & (self.score_data['Label'] == 1.0), mod].tolist()
            imps = self.score_data.loc[(self.score_data['Train_Test'] == 'TRAIN') & (self.score_data['Label'] == 0.0), mod].tolist()

            self.make_density_plots(gen=gens, imp=imps,
                                    label='Training', norm_type=self.normalize, modality=mod, exp=exp_id)

            # Testing Data
            gens = self.score_data.loc[(self.score_data['Train_Test'] == 'TEST') & (self.score_data['Label'] == 1.0), mod].tolist()
            imps = self.score_data.loc[(self.score_data['Train_Test'] == 'TEST') & (self.score_data['Label'] == 0.0), mod].tolist()

            self.make_density_plots(gen=gens, imp=imps,
                                    label='Testing', norm_type=self.normalize, modality=mod, exp=exp_id)

            # All Data
            self.make_density_plots(gen=self.score_data.loc[self.score_data['Label'] == 1.0, mod],
                                    imp=self.score_data.loc[self.score_data['Label'] == 0.0, mod],
                                    label='Entire', norm_type=self.normalize, modality=mod, exp=exp_id)

    def get_modalities(self):
        return self.modalities

    def get_score_data(self):
        return self.score_data

    def get_beans(self):
        imps_train = len(self.score_data[(self.score_data['Train_Test'] == 'TRAIN') & (self.score_data['Label'] == 0.0)].index)
        imps_test = len(self.score_data[(self.score_data['Train_Test'] == 'TEST') & (self.score_data['Label'] == 0.0)].index)

        gen_train = len(self.score_data[(self.score_data['Train_Test'] == 'TRAIN') & (self.score_data['Label'] == 1.0)].index)
        gen_test = len(self.score_data[(self.score_data['Train_Test'] == 'TEST') & (self.score_data['Label'] == 1.0)].index)

        return {'imp_train': imps_train, 'imp_test': imps_test,
                'gen_train': gen_train, 'gen_test': gen_test,}


    def update_datas(self, change_dics):
        original = list(change_dics.keys())

        # Name Changes
        updated_names = []
        dissimilarity = []
        similarity = []
        dont_use = []

        for key, value in change_dics.items():
            updated_names.append(value[0])
            if value[1]:
                similarity.append(value[0])

            if value[2]:
                dissimilarity.append(value[0])

            if not value[3]:
                dont_use.append(value[0])

        # Name Changes
        for i in range(len(updated_names)):
            key = original[i]
            new_name = updated_names[i]

            self.score_data.rename(columns={key: new_name}, inplace=True)

        self.modalities = [x for x in self.score_data.columns if x not in ['Label', 'Probe_ID', 'Gallery_ID', 'Train_Test'] and '_ORIGINAL' not in x]

        # Dissimilarity Changes
        for mod in dissimilarity:
            self.score_data[mod] = 1 - self.score_data[mod]

        # Remove modalitiy
        for mod in dont_use:
            self.score_data.drop(mod, axis=1, inplace=True)
            self.modalities.remove(mod)