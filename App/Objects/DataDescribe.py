import os, shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataDescribe:
    def __init__(self, **kwargs):
        self.sparcity = None
        self.modals = kwargs.get('modals')
        self.data = kwargs.get('df')
        self.train = self.data[self.data['TRAIN_TEST'] == 'TRAIN']
        self.test = self.data[self.data['TRAIN_TEST'] == 'TEST']

        self.beans = pd.DataFrame(index=['Dataset', 'Train-Set', 'Test-Set'],
                                  columns=['Total_Subjects', 'Genuine_Subjects', 'Imposter_Subjects',
                                           'Total_Probes', 'Genuine_Probes', 'Imposter_Probes'])

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
        sparcity = pd.DataFrame(columns=['Probes', '% Full',
                                         'Probes_gen', '% Full_gen',
                                         'Probes_imp', '% Full_imp', 'Split'])

        splits = ['Entire', 'Train', 'Test']
        sparcity.at['Total', '% Full'] = len(self.data.dropna()) / len(self.data)

        for s in splits:
            if s == 'Test':
                df = self.test
            elif s == 'Train':
                df = self.train
            else:
                df = self.data

            for m in self.modals:
                tmp = df[m]
                sparcity.at[m + '-' + s, 'Split'] = s
                sparcity.at[m + '-' + s, 'Probes'] = len(tmp.dropna())
                sparcity.at[m + '-' + s, '% Full'] = len(tmp.dropna()) / len(tmp)

                tmp_gen = df[df['LABEL'] == 1.0]
                sparcity.at[m + '-' + s, 'Probes_gen'] = len(tmp_gen.dropna())
                sparcity.at[m + '-' + s, '% Full_gen'] = len(tmp_gen.dropna()) / len(tmp_gen)

                tmp_imp = df[df['LABEL'] == 0.0]
                sparcity.at[m + '-' + s, 'Probes_imp'] = len(tmp_imp.dropna())
                sparcity.at[m + '-' + s, '% Full_imp'] = len(tmp_imp.dropna()) / len(tmp_imp)

        sparcity.to_pickle('./generated/sparcity_beans.pk')
        self.sparcity = sparcity

    def update_test(self, train, test):
        self.train = train
        self.test = test

    @staticmethod
    def reset_density_plots():
        cats = ['overlap', 'PDF', 'hist']

        for c in cats:
            folder = './generated/density/' + c + '/'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def make_density_plots(self, subset='Test'):
        if not os.path.exists('./generated/density/PDF/' + subset + '/'):
            os.makedirs('./generated/density/PDF/' + subset + '/')
        if not os.path.exists('./generated/density/hist/' + subset + '/'):
            os.makedirs('./generated/density/hist/' + subset + '/')
        if not os.path.exists('./generated/density/overlap/' + subset + '/'):
            os.makedirs('./generated/density/overlap/' + subset + '/')

        if subset.lower() == 'test':
            analysis_set = self.test
        elif subset.lower() == 'train':
            analysis_set = self.train
        else:
            analysis_set = self.data

        x_min = 0.0
        x_max = 1.0

        for m in self.modals:
            if x_min < analysis_set[m].min():
                x_min = analysis_set[m].min()
            if x_max > analysis_set[m].max():
                x_max = analysis_set[m].max()

            gen = analysis_set[analysis_set['LABEL'] == 1.0][m]
            imp = analysis_set[analysis_set['LABEL'] == 0.0][m]

            fig = plt.figure(figsize=(7, 6))
            #################################################################
            # pdf
            #################################################################
            sns.kdeplot(imp, fill=True, label='Imposter (pdf) ', color='#C89A58')
            sns.kdeplot(gen, fill=True, label='Genuine (pdf) ', color='#0DB14B')

            leg = plt.legend(loc=2)
            plt.ylabel(r"Density Estimate")
            ax = plt.gca()
            # lims = ax.get_xlim()
            # ax.set_xlim(lims)
            plt.xlim([x_min, x_max])
            # plt.xlim([0.0, 1.0])
            ax.set_xlabel(m, fontsize=20, fontweight='bold')
            fig.savefig(
                './generated/density/PDF/' + subset + '/' + m + '.png')
            leg.remove()

            #################################################################
            # Overlaid
            #################################################################
            ax2 = plt.twinx()
            sns.histplot(imp, kde=False, label='Imposter (hist)', color='#FF1493')
            sns.histplot(gen, kde=False, label='Genuine (hist)', color='#7B68EE')

            fig.legend(loc=2, borderaxespad=3)
            ax.set_ylabel(r"Density Estimate")
            ax2.set_ylabel(r"Sample Counts")

            ax.set_xlabel(m, fontsize=20, fontweight='bold')
            plt.xlim([x_min, x_max])

            fig.savefig('./generated/density/overlap/' + subset + '/' + m + '.png')
            plt.clf()

            #################################################################
            # histogram
            #################################################################
            sns.histplot(imp, kde=False, label='Imposter (hist)', color='#FF1493')
            sns.histplot(gen, kde=False, label='Genuine (hist)', color='#7B68EE')
            ax = plt.gca()

            plt.legend(loc=2)
            ax.set_xlabel(m, fontsize=20, fontweight='bold')
            ax.set_ylabel("Sample Counts")
            plt.xlim([x_min, x_max])

            ax_c = ax.twinx()
            ax_c.set_ylabel('')

            fig.savefig('./generated/density/hist/' + subset + '/' + m + '.png')
            plt.clf()
            plt.close('all')
