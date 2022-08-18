import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder

# Custom Imports

# styles
kivy.require('2.0.0')
Builder.load_file('kvs/density.kv')


class Densities(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.imputation = None
        # self.beans = None

    def set_beans(self, counts, modalities):
        print(counts.to_markdown())
        # self.ids['num_training_subs'].text = '[b]{:,} Probe Subjects[/b]'.format(counts.at['Train-Set',
        #                                                                                    'Genuine_Subjects'])
        #
        # self.ids['num_testing_subs'].text = '[b]{:,} Probe Subjects[/b]'.format(counts.at['Test-Set',
        #                                                                                   'Genuine_Subjects'])

        self.ids['num_training'].text = '[b]Probes: [/b] {:,} '.format(counts.at['Train-Set',
                                                                                 'Total_Probes'])
        self.ids['num_training_gens'].text = '[b]Genuine Probes: [/b] {:,} '.format(counts.at['Train-Set',
                                                                                              'Genuine_Probes'])
        self.ids['num_training_imps'].text = '[b]Imposter Probes: [/b] {:,} '.format(counts.at['Train-Set',
                                                                                               'Imposter_Probes'])

        self.ids['num_testing'].text = '[b]Probes: [/b] {:,} '.format(counts.at['Test-Set',
                                                                                'Total_Probes'])
        self.ids['num_testing_gens'].text = '[b]Genuine Probes: [/b] {:,} '.format(counts.at['Test-Set',
                                                                                             'Genuine_Probes'])
        self.ids['num_testing_imps'].text = '[b]Imposter Probes: [/b] {:,} '.format(counts.at['Test-Set',
                                                                                              'Imposter_Probes'])

        self.ids['num_mods'].text = str(len(modalities))
        self.ids['num_probes'].text = '[b]{:,}[/b]'.format(counts.at['Dataset', 'Total_Probes'])