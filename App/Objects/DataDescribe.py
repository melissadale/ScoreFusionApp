import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataDescribe:
    def __init__(self, **kwargs):
        self.modals = kwargs.get('modals')
        self.train = kwargs.get('train')
        self.test = kwargs.get('test')
        self.data = kwargs.get('df')

        self.beans = pd.DataFrame(index=['Dataset', 'Train-Set', 'Test-Set'],
                                  columns=['Total_Subjects', 'Genuine_Subjects', 'Imposter_Subjects',
                                           'Total_Probes', 'Genuine_Probes', 'Imposter_Probes'])

        self.sparcity = pd.DataFrame(index=[m for m in self.modals],
                                     columns=['Probes', '% Full',
                                              'Probes_gen', '% Full_gen',
                                              'Probes_imp', '% Full_imp'])

        self.title_cleanup = {
            'Score_oc_right': 'Right Ocular',
            'Score_oc_left': 'Left Ocular',
            'Score_iris_right': 'Right Iris',
            'Score_iris_left': 'Left Iris',
        }

    def count_beans(self):
        self.beans.at['Dataset', 'Total_Subjects'] = len(self.data['PROBE_ID'].unique())
        self.beans.at['Dataset', 'Genuine_Subjects'] = len(self.data[self.data['LABEL'] == 1.0]['PROBE_ID'].unique())
        self.beans.at['Dataset', 'Imposter_Subjects'] = len(self.data[self.data['LABEL'] == 0.0]['PROBE_ID'].unique())

        self.beans.at['Train-Set', 'Total_Subjects'] = len(self.train['PROBE_ID'].unique())
        self.beans.at['Train-Set', 'Genuine_Subjects'] = len(
            self.train[self.train['LABEL'] == 1.0]['PROBE_ID'].unique())
        self.beans.at['Train-Set', 'Imposter_Subjects'] = len(
            self.train[self.train['LABEL'] == 0.0]['PROBE_ID'].unique())

        self.beans.at['Test-Set', 'Total_Subjects'] = len(self.test['PROBE_ID'].unique())
        self.beans.at['Test-Set', 'Genuine_Subjects'] = len(self.test[self.test['LABEL'] == 1.0]['PROBE_ID'].unique())
        self.beans.at['Test-Set', 'Imposter_Subjects'] = len(self.test[self.test['LABEL'] == 0.0]['PROBE_ID'].unique())

        self.beans.at['Dataset', 'Total_Probes'] = len(self.data)
        self.beans.at['Dataset', 'Genuine_Probes'] = len(self.data[self.data['LABEL'] == 1.0])
        self.beans.at['Dataset', 'Imposter_Probes'] = len(self.data[self.data['LABEL'] == 0.0])

        self.beans.at['Train-Set', 'Total_Probes'] = len(self.train)
        self.beans.at['Train-Set', 'Genuine_Probes'] = len(self.train[self.train['LABEL'] == 1.0])
        self.beans.at['Train-Set', 'Imposter_Probes'] = len(self.train[self.train['LABEL'] == 0.0])

        self.beans.at['Test-Set', 'Total_Probes'] = len(self.test)
        self.beans.at['Test-Set', 'Genuine_Probes'] = len(self.test[self.test['LABEL'] == 1.0])
        self.beans.at['Test-Set', 'Imposter_Probes'] = len(self.test[self.test['LABEL'] == 0.0])

    def modality_counts(self):
        for m in self.modals:
            tmp = self.data[[m, 'Label']]
            tmp.dropna(inplace=True, axis=0)

            print(len(tmp))
            print(len(tmp[self.data['Label'] == 1.0]))
            print(len(tmp[self.data['Label'] == 0.0]))

    def print_eval(self, style='markdown'):
        if style.lower() == 'markdown':
            print(self.beans.to_markdown())
        if style.lower() == 'csv':
            self.beans.to_csv('./Results/DataDescribe.csv')

    # def sparcenessness(self):
    #     t = self.data[self.modals].isnull().sum(axis=1).value_counts()
    #
    #     for m in self.modals:
    #         # r = df[df[modality].notnull()].index.tolist()
    #         self.sparcity.at[m, 'Total_Subjects']
    #         tmp = self.data[m]
    #         # missing = sum(tmp.apply(lambda x: sum(x.isnull().values), axis=1) > 0)
    #         # print('Missing: ' + str(missing))
    #         print('Not Missing: ' + str(len(tmp.dropna())))
    #         print('prec : '+ str(len(tmp.dropna())/len(tmp)))

    def make_density_plots(self, subset='Test'):
        if not os.path.exists('./generated/density/PDF/'):
            os.makedirs('./generated/density/PDF/')
        if not os.path.exists('./generated/density/hist/'):
            os.makedirs('./generated/density/hist/')
        if not os.path.exists('./generated/density/overlap/'):
            os.makedirs('./generated/density/overlap/')

        if subset.lower() == 'test':
            analysis_set = self.test
        elif subset.lower() == 'train':
            analysis_set = self.train
        else:
            analysis_set = self.data

        for m in self.modals:
            gen = analysis_set[analysis_set['LABEL'] == 1.0][m]
            imp = analysis_set[analysis_set['LABEL'] == 0.0][m]

            #################################################################
            # Overlaid
            #################################################################
            sns.kdeplot(imp, fill=True, label='Imposter', color='#C89A58')
            sns.kdeplot(gen, fill=True, label='Genuine', color='#0DB14B')
            ax = plt.gca()
            ax2 = plt.twinx()
            sns.histplot(imp, kde=False, label='Imposter', color='#FF1493')
            sns.histplot(gen, kde=False, label='Genuine', color='#7B68EE')

            p = 'Density Estimates and Score Counts for ' + m + '\n' + str(len(gen)) + ' Subjects '

            ax2.legend(bbox_to_anchor=(1, 1), loc='upper center')
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            lims = ax.get_xlim()
            y_ticks = ax.get_yticks()
            ax.set_ylabel(r"Density Estimate")
            ax2.set_ylabel(r"Sample Counts")

            ax.set_xlabel(m)
            plt.title(p)

            # plt.savefig('./generated/density/overlap/' + m + '-' + subset + '.png', bbox_inches='tight')
            plt.clf()

            #################################################################
            # pdf
            #################################################################
            sns.kdeplot(imp, fill=True, label='Imposter', color='#C89A58')
            sns.kdeplot(gen, fill=True, label='Genuine', color='#0DB14B')

            p = 'Density Estimates for ' + m + '\n ' + str(len(gen)) + ' Subjects '
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            plt.ylabel(r"Density Estimate")
            plt.title(p)
            ax = plt.gca()
            ax.set_xlim(lims)
            ax.set_xlabel(m)
            # plt.savefig(
            #     './generated/density/PDF/' + m + '-' + subset + '.png',
            #     bbox_inches='tight')
            plt.clf()

            #################################################################
            # histogram
            #################################################################

            sns.histplot(imp, kde=False, label='Imposter', color='#FF1493')
            sns.histplot(gen, kde=False, label='Genuine', color='#7B68EE')
            ax = plt.gca()

            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            p = 'Score counts for ' + m + '\n' + str(len(gen)) + ' Subjects '

            ax.set_xlim(lims)
            ax.set_yticks(y_ticks)

            plt.title(p)
            ax.set_ylabel(r"Density Estimate")
            ax2.set_ylabel(r"Sample Counts")

            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='y', colors='white')

            ax.set_xlabel(m)

            # plt.savefig('./generated/density/hist/' + m + '-' + subset + '.png', bbox_inches='tight')
            plt.clf()
