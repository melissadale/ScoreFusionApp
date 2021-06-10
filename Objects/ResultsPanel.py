from Objects.ROCsSlider import ROCsPlots

class Results:
    def __init__(self, **kwargs):
        self.active = 'ROC'
        self.toggle = './graphics/ROC.png'
        self.active_display_path = ''

        # Right Panel Info
        self.top_header = ''
        self.middle_header = ''
        self.bottom_header = ''

        self.roc_object = ROCsPlots(slider=kwargs.get('slider'), experiment=kwargs.get('experiment'))

    def change_setting(self):
        if self.active == 'ROC':
            self.active = 'CMC'
            self.toggle = './graphics/CMC.png'
            self.active_display_path = self.roc_object.build_plot_list()

        elif self.active == 'CMC':
            self.active = 'ROC'
            self.toggle = './graphics/ROC.png'


    def build_plot_list(self):
        if self.active == 'ROC':
            return self.roc_object.build_plot_list()

    def update_plot(self):
        if self.active == 'ROC':
            return self.roc_object.update_plot()

    def slider_update(self):
        if self.active == 'ROC':
            return self.roc_object.get_experiment()


    def move_left(self):
        if self.active == 'ROC':
            return self.roc_object.move_left()

    def move_right(self):
        if self.active == 'ROC':
            return self.roc_object.move_right()

    def get_top_header(self):
        return self.top_header


    def get_plot(self):
        return self.active_display_path

    def get_experiment(self):
        if self.active == 'ROC':
            return self.roc_object.get_experiment()

    def get_toggle(self):
        return self.toggle