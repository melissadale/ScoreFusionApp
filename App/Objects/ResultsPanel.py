from Objects.ROCsSlider import ROCsPlots
from Objects.CMCsSlider import CMCsPlots

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
        self.cmc_object = CMCsPlots(slider=kwargs.get('slider'), experiment=kwargs.get('experiment'))

    def change_setting(self, given=False):
        if given:
            if given == 'ROC':
                self.active = 'ROC'
                self.build_plot_list()
                self.toggle = './graphics/ROC.png'
                self.active_display_path = './generated/experiments/ROC/' + self.get_experiment() + '/CMC-all.png'

            elif given == 'CMC':
                self.active = 'CMC'
                self.build_plot_list()
                self.toggle = './graphics/CMC.png'
                self.active_display_path = './generated/experiments/CMC/' + self.get_experiment() + '/all.png'
        else:
            if self.active == 'ROC':
                self.active = 'CMC'
                self.build_plot_list()
                self.toggle = './graphics/CMC.png'
                self.active_display_path = './generated/experiments/CMC/' + self.get_experiment() + '/CMC-all.png'

            elif self.active == 'CMC':
                self.active = 'ROC'
                self.build_plot_list()
                self.toggle = './graphics/ROC.png'
                self.active_display_path = './generated/experiments/ROC/' + self.get_experiment() + '/all.png'

        return self.active_display_path

    def build_plot_list(self):
        if self.active == 'ROC':
            return self.roc_object.build_plot_list()
        if self.active == 'CMC':
            return self.cmc_object.build_plot_list()

    def update_plot(self):
        if self.active == 'ROC':
            return self.roc_object.update_plot()
        if self.active == 'CMC':
            return self.cmc_object.update_plot()

    def slider_update(self):
        if self.active == 'ROC':
            return self.roc_object.get_experiment()
        if self.active == 'CMC':
            return self.cmc_object.get_experiment()


    def move_left(self):
        if self.active == 'ROC':
            return self.roc_object.move_left()
        if self.active == 'CMC':
            return self.cmc_object.move_left()

    def move_right(self):
        if self.active == 'ROC':
            return self.roc_object.move_right()
        if self.active == 'CMC':
            return self.cmc_object.move_right()

    def get_top_header(self):
        return self.top_header


    def get_plot(self):
        return self.active_display_path

    def get_experiment(self):
        if self.active == 'ROC':
            return self.roc_object.get_experiment()
        if self.active == 'CMC':
            return self.cmc_object.get_experiment()

    def get_toggle(self):
        return self.toggle

    def get_active(self):
        return self.active

    def get_headers(self):
        if self.active == 'ROC':
            return 'AUC Accuracy', 'Equal Error Rate', 'Estimated TMR', '@ Fixed FMR', '0.01'
        elif self.active == 'CMC':
            return 'Rank 1 Accuracy', 'Rank 2 Accuracy', 'Rank K Accuracy', 'K = ', '5'