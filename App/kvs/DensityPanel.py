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
        self.display_path_density = '../generated/density/overlap/Train/face-C.png'

    def set_beans(self, counts, sparcity, modalities):
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

        self.ids['num_mods'].text = '[b]' + str(len(modalities)) + '[/b]'
        self.ids['num_probes'].text = '[b]{:,}[/b]'.format(counts.at['Dataset', 'Total_Probes'])
        self.ids['num_subs'].text = '[b]{:,}[/b]'.format(counts.at['Dataset', 'Total_Subjects'])
        self.ids['perc_full'].text = '[b]{:.0%}[/b]'.format(sparcity.at['Total', '% Full'])
    def update_density_type(self):
        print('Press')