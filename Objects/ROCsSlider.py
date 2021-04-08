import glob
import os


class ROCsPlots:
    def __init__(self, **kwargs):

        self.plots_list = []
        self.current_plot_index = 0
        self.current_plot_path = ''

        self.slider = kwargs.get('slider')
        self.experiment_ids = next(os.walk('./generated/experiments/'))[1]
        self.number_experiments = len(self.experiment_ids)
        self.experiment_pointer = self.number_experiments-1

    def update_plot(self):
        self.current_plot_index = (self.current_plot_index + 1) % len(self.plots_list)
        self.current_plot_path = self.plots_list[self.current_plot_index]
        return self.get_image_path()

    def slider_update(self, val):
        self.experiment_pointer = val-1
        return self.get_image_path()

    def move_left(self):
        if self.experiment_pointer >= 0:
            self.experiment_pointer = self.experiment_pointer-1
            self.slider.value = self.slider.value - 1
        return self.get_image_path()

    def move_right(self):
        if self.current_plot_index < self.number_experiments-1:
            self.experiment_pointer = self.experiment_pointer+1
            self.slider.value = self.slider.value + 1
        return self.get_image_path()

    def build_plot_list(self):
        plots = []
        base = './generated/experiments/' + self.experiment_ids[self.experiment_pointer] + '/'

        for filename in glob.glob(base + '/*.png'):
            plots.append(filename.replace('./generated/experiments/' + self.experiment_ids[self.experiment_pointer] , ''))

        self.slider.max = self.number_experiments
        self.plots_list = plots
        return self.get_image_path()

    def get_image_path(self):
        return './generated/experiments/' + self.experiment_ids[self.experiment_pointer] + '/' + self.plots_list[self.current_plot_index]

    def get_experiment(self):
        return self.experiment_ids[self.experiment_pointer]