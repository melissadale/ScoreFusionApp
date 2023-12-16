import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.clock import Clock
import functools

# Custom Imports
from kvs.ImputationPopup import Imputation as ImputationPop
from kvs.NormalizationPopups.DoubleSigmoidPopup import DSigPopup
from kvs.NormalizationPopups.TanhPopup import TanhPopup
from Objects.ScoreData import ScoreData
from kvs.EditModalityPopup import ModeEditPopup

# styles
kivy.require('2.0.0')
Builder.load_file('kvs/data.kv')
Builder.load_file('kvs/Impute.kv')
Builder.load_file('kvs/NormalizationPopups/DoubleSigmoid.kv')
Builder.load_file('kvs/NormalizationPopups/Tanh.kv')
Builder.load_file('kvs/ModalityEdit.kv')


class Data(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.edit_mods = None
        self.beans = None
        self.imputation = None
        self.score_data = ScoreData()
        self.train_perc = 80
        self.ids['test_label'].text = '   % Testing - ' + str(self.train_perc) + '% Training \n'
        self.ids['data_path'].text = ''
        self.normalize = 'MinMax'
        self.norm_params = None

    def impute_click(self, instance, value):
        if value is True:
            self.impute_popup()
            self.ids['missing_ignore_chk'].active = False

    def ignore_click(self, instance, value):
        if value is True:
            self.ids['missing_impute_chk'].active = False

    def set_data_location(self, path):
        self.ids['data_path'].text = path

    def get_data_location(self):
        return self.data_path

    # Popups
    def DSig_popup(self):
        show_dsig = DSigPopup()
        dsig_popup = Popup(title="Double Sigmoid Estimator ", content=show_dsig, size_hint=(None, None),
                           size=(400, 400))
        dsig_popup.open()  # show the popup
        show_dsig.set_pop(dsig_popup)
        self.norm_params = show_dsig

    def Tanh_popup(self):
        show_tanh = TanhPopup()
        tanh_popup = Popup(title="Tanh Estimator", content=show_tanh, size_hint=(None, None),
                           size=(400, 400))
        tanh_popup.open()  # show the popup
        show_tanh.set_pop(tanh_popup)
        self.norm_params = show_tanh

    def modality_edit_popup(self):
        emod = ModeEditPopup(modality_list=self.score_data.get_modalities())
        popup = Popup(title="Edit Modalities", content=emod, size_hint=(None, None),
                      size=(600, 600))
        popup.open()
        emod.set_pop(popup)
        self.edit_mods = emod

    def impute_popup(self):
        imputation_settings = ImputationPop()
        popup = Popup(title="Imputation", content=imputation_settings, size_hint=(None, None),
                      size=(600, 600))
        imputation_settings.set_pop(popup)
        popup.open()
        self.imputation = imputation_settings

    # Messages and GUI changes
    def update_test(self, test_p):
        self.train_perc = 100 - test_p
        self.ids['test_label'].text = '   % Testing - ' + str(self.train_perc) + '% Training \n'

    def train_split_checkbox(self, instance, value, category='train-test'):
        if category == 'train-test':
            if value is True:
                self.ids['load_previous_chk'].active = False
                self.ids['divide_data'].opacity = 1.0
                self.ids['load_previous'].opacity = 0.0

            else:
                self.ids['divide_data'].opacity = 0.0

        elif category == 'previous':
            if value is True:
                self.ids['train_test_chk'].active = False
                self.ids['load_previous'].opacity = 1.0
                self.ids['divide_data'].opacity = 0.0

            else:
                self.ids['load_previous'].opacity = 0.0

    def update_bar(self, params, dt):
        if self.ids['load_pb'].value <= 100:
            self.ids['load_pb'].value += params[0]
            self.ids['pb_status_lbl'].text = params[1]

    # Operations
    def get_data_files(self):
        if self.ids['train_test_chk'].active:
            training = self.train_perc
        else:
            training = False

        # Load data and split into train/test
        Clock.schedule_once(functools.partial(self.update_bar, [5, 'Loading Data ...']))

        self.score_data.load_data(self.ids['data_path'].text, training)
        Clock.schedule_once(functools.partial(self.update_bar, [25, 'Normalizing Data ...']))

        # Normalize Data
        if self.norm_params is not None:
            params = self.norm_params.get_params()
        else:
            params = None

        self.score_data.normalize_scores(self.normalize, params)
        Clock.schedule_once(functools.partial(self.update_bar, [25, 'Imputing Missing ...']))

        # Perform Imputations
        if self.imputation:
            self.score_data.impute(self.imputation.get_imputation())
        Clock.schedule_once(functools.partial(self.update_bar, [25, 'Visualizing Data ...']))

        # Visualize Data
        modalities = self.score_data.get_modalities()
        remaining_pb = int(10 / len(modalities))
        for mod in modalities:
            Clock.schedule_once(functools.partial(self.update_bar, [remaining_pb, 'Visualizing and Collecting '
                                                                                  'Metrics ...']))
            self.ids['modalities_lbl'].text = self.ids['modalities_lbl'].text + '\n\n' + mod

        self.beans = self.score_data.get_descripts() # todo: save sparcity to pickle
        Clock.schedule_once(
            functools.partial(self.update_bar, [100 - self.ids['load_pb'].value, 'Done Processing Input']))

        self.ids['detected_mods_btn'].background_normal = './kvs/graphics/pencil.png'
        self.ids['detected_mods_btn'].background_down = './kvs/graphics/pencil-grey.png'
        # refresh_btn
        self.ids['refresh_btn'].color = (0.05, 0.69, 0.29, 1)

    def update_test_msg(self):
        self.ids['test_label'].text = '% Testing, ' + str(100 - int(self.ids['test_input'].text)) + "% Training"

    def refresh(self):
        try:
            changes = self.edit_mods.get_updates()

            self.ids['modalities_lbl'].text = changes['new_msg']

            # reset loading bar
            Clock.schedule_once(functools.partial(self.update_bar,
                                                  [-50, 'ReVisualizing and ReCollecting ...']))

            for key, item in changes.items():
                if key == 'new_msg':
                    continue

                if not item[3]:  # Do Not Use this modality
                    self.score_data.data.drop([key], axis=1, inplace=True)
                    continue

                if item[2]:  # Dissimilarity Score
                    self.score_data.data[key] = 1 - self.score_data.data[key]

                # rename
                self.score_data.data.rename({key: item[0]}, inplace=True)

            self.score_data.get_descripts(reset=True)
            Clock.schedule_once(functools.partial(self.update_bar,
                                                  [50, 'Done Updating Input']))

        except:
            pass
