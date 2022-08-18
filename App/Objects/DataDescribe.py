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

        self.sparcity = pd.DataFrame(index=[m for m in self.modals] + ['Total'],
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

    def print_eval(self, style='markdown'):
        if style.lower() == 'markdown':
            print(self.beans.to_markdown())
        if style.lower() == 'csv':
            self.beans.to_csv('./Results/DataDescribe.csv')

    def sparcenessness(self):
        self.sparcity.at['Total', '% Full'] = len(self.data.dropna()) / len(self.data)

        for m in self.modals:
            tmp = self.data[m]
            self.sparcity.at[m, 'Probes'] = len(tmp.dropna())
            self.sparcity.at[m, '% Full'] = len(tmp.dropna()) / len(tmp)

            tmp_gen = self.data[self.data['LABEL'] == 1.0]
            self.sparcity.at[m, 'Probes_gen'] = len(tmp_gen.dropna())
            self.sparcity.at[m, '% Full_gen'] = len(tmp_gen.dropna()) / len(tmp_gen)

            tmp_imp = self.data[self.data['LABEL'] == 0.0]
            self.sparcity.at[m, 'Probes_imp'] = len(tmp_imp.dropna())
            self.sparcity.at[m, '% Full_imp'] = len(tmp_imp.dropna()) / len(tmp_imp)

    def make_density_plots(self, subset='Test'):
        if not os.path.exists('./generated/density/PDF/'+subset+'/'):
            os.makedirs('./generated/density/PDF/'+subset+'/')
        if not os.path.exists('./generated/density/hist/'+subset+'/'):
            os.makedirs('./generated/density/hist/'+subset+'/')
        if not os.path.exists('./generated/density/overlap/'+subset+'/'):
            os.makedirs('./generated/density/overlap/'+subset+'/')

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
            fig = plt.figure(figsize=(6, 6))
            sns.kdeplot(imp, fill=True, label='Imposter', color='#C89A58')
            sns.kdeplot(gen, fill=True, label='Genuine', color='#0DB14B')
            ax = plt.gca()
            ax2 = plt.twinx()
            sns.histplot(imp, kde=False, label='Imposter', color='#FF1493')
            sns.histplot(gen, kde=False, label='Genuine', color='#7B68EE')

            ax2.legend(bbox_to_anchor=(1, 1), loc='upper center')
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            lims = ax.get_xlim()
            y_ticks = ax.get_yticks()
            ax.set_ylabel(r"Density Estimate")
            ax2.set_ylabel(r"Sample Counts")

            ax.set_xlabel(m, fontsize=20, fontweight='bold')

            fig.savefig('./generated/density/overlap/' + subset + '/' + m + '.png', bbox_inches='tight')
            plt.clf()

            #################################################################
            # pdf
            #################################################################
            sns.kdeplot(imp, fill=True, label='Imposter', color='#C89A58')
            sns.kdeplot(gen, fill=True, label='Genuine', color='#0DB14B')

            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            plt.ylabel(r"Density Estimate")
            ax = plt.gca()
            ax.set_xlim(lims)
            ax.set_xlabel(m, fontsize=20, fontweight='bold')
            fig.savefig(
                './generated/density/PDF/' + subset + '/' + m + '.png',
                bbox_inches='tight')
            plt.clf()

            #################################################################
            # histogram
            #################################################################
            sns.histplot(imp, kde=False, label='Imposter', color='#FF1493')
            sns.histplot(gen, kde=False, label='Genuine', color='#7B68EE')
            ax = plt.gca()

            plt.legend(bbox_to_anchor=(1, 1), loc=2)

            ax.set_xlim(lims)
            ax.set_yticks(y_ticks)

            ax.set_ylabel(r"Density Estimate")
            ax2.set_ylabel(r"Sample Counts")

            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='y', colors='white')

            ax.set_xlabel(m, fontsize=20, fontweight='bold')

            fig.savefig('./generated/density/hist/' + subset + '/' + m + '.png', bbox_inches='tight')
            plt.clf()
