import glob
import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder

# styles
kivy.require('2.0.0')
Builder.load_file('kvs/density.kv')


class DensityPlots:
    def __init__(self, **kwargs):
        self.plot_type = ['overlap', 'PDF', 'hist']
        self.plot_type_pointer = 0

        self.train_test_grouping = 'Entire'
        self.plots_list = []
        self.current_plot_index = 0

        self.current_plot_path = ''

        self.density_slider = kwargs.get('slider')

    def update_plot_type(self):
        self.plot_type_pointer = (self.plot_type_pointer + 1) % 3
        self.current_plot_path = self.plot_type[self.plot_type_pointer]
        self.build_plot_list()

        return self.get_image_path()

    def update_test_train(self, group):
        self.train_test_grouping = group

        self.build_plot_list()
        return self.get_image_path()

    def update_plot(self, value):
        self.current_plot_index = value
        self.current_plot_path = self.plots_list[value]
        return self.get_image_path()

    def move_left(self):
        if self.current_plot_index > 0:
            self.current_plot_index = self.current_plot_index - 1
            self.density_slider.value = self.density_slider.value - 1
            return self.update_plot(self.current_plot_index)
        else:
            return self.current_plot_path

    def move_right(self):
        if self.current_plot_index < self.density_slider.max - 1:
            self.current_plot_index = self.current_plot_index + 1
            self.density_slider.value = self.density_slider.value + 1
            return self.update_plot(self.current_plot_index)
        else:
            return self.current_plot_path

    def build_plot_list(self):
        plots = []
        base = './generated/density/' + self.plot_type[self.plot_type_pointer] + \
               '/' + self.train_test_grouping

        for filename in glob.glob(base + '/*.png'):
            plots.append(filename)

        self.density_slider.max = len(plots)
        self.plots_list = plots
        self.current_plot_path = plots[self.current_plot_index]

    def get_image_path(self):
        return self.current_plot_path

    def get_numb_plots(self):
        return len(self.plots_list)


class Densities(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.densities_plots = DensityPlots(slider=self.ids.density_slider_id)
        self.num_mods_slider = 0

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
        self.num_mods_slider = len(modalities)

    def set_paths(self):
        self.densities_plots.build_plot_list()
        self.ids['density_button'].current_path = self.densities_plots.get_image_path()

    def density_slider(self, value):
        if 0 <= value < self.ids.density_slider_id.max + 1:
            self.ids['density_button'].current_path = self.densities_plots.update_plot(value)

    def update_density_type(self):
        self.densities_plots.update_plot_type()
        self.ids['density_button'].current_path = self.densities_plots.get_image_path()

    def set_density_set(self):
        if self.ids.chk_Test.active:
            self.ids['density_button'].current_path = self.densities_plots.update_test_train('Test')
        elif self.ids.chk_Train.active:
            self.ids['density_button'].current_path = self.densities_plots.update_test_train('Train')
        else:
            self.ids['density_button'].current_path = self.densities_plots.update_test_train('Entire')

    def slider_button(self, direction, value):
        if direction == 'left':
            self.ids['density_button'].current_path = self.densities_plots.move_left()

        if direction == 'right':
            self.ids['density_button'].current_path = self.densities_plots.move_right()
