import glob


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
            self.current_plot_index = self.current_plot_index-1
            self.density_slider.value = self.density_slider.value - 1
            return self.update_plot(self.current_plot_index)
        else:
            return self.current_plot_path

    def move_right(self):
        if self.current_plot_index < len(self.plots_list)-1:
            self.current_plot_index = self.current_plot_index+1
            self.density_slider.value = self.density_slider.value + 1
            return self.update_plot(self.current_plot_index)
        else:
            return self.current_plot_path

    def build_plot_list(self):
        plots = []
        base = './generated/density/' + self.train_test_grouping + \
               '/' + self.plot_type[self.plot_type_pointer]

        for filename in glob.glob(base + '/*.png'):
            plots.append(filename)

        self.density_slider.max = len(plots)-1
        self.plots_list = plots
        self.current_plot_path = plots[self.current_plot_index]

    def get_image_path(self):
        return self.current_plot_path

    def get_numb_plots(self):
        return len(self.plots_list)
