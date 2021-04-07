"""
Created on 1/29/2020
By Melissa Dale
"""
import glob
import os
import shutil

# os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'

import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial
from numpy import ones, vstack
from numpy.linalg import lstsq

import Analytics.format_data as fm2
import Analytics.Fuse as Fuse
import Popups.PopupSave as SavePop
import Popups.PopupReset as ResetPopup
import Popups.PopupTanh as TanhPopup
import Popups.PopupDSig as DSigPopup
import Popups.PopupSelectiveFusion as SelectiveFusionPopup
import Popups.PopupModalityEdit as PopupModalityEdit
from Objects.ReportPDFs import generate_summary
from Objects.DensitySlider import DensityPlots
from Objects.ROCsSlider import ROCsPlots

# Program to explain how to create tabbed panel App in kivy: https://www.geeksforgeeks.org/python-tabbed-panel-in-kivy/
import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider

kivy.require('1.9.0')


class ScreenManagement(ScreenManager):
    pass


class Loc(Screen):
    load_location = StringProperty('')
    data = ObjectProperty(None)

    def __init__(self, **args):
        Clock.schedule_once(self.init_widget, 0)
        return super(Loc, self).__init__(**args)

    def init_widget(self, *args):
        fc = self.ids['filechoose']
        fc.bind(on_entry_added=self.update_file_list_entry)
        fc.bind(on_subentry_to_entry=self.update_file_list_entry)

    def update_file_list_entry(self, file_choose, file_list_entry, *args):
        file_list_entry.children[0].color = (0.0, 0.0, 0.0, 1.0)  # File Names
        file_list_entry.children[1].color = (0.0, 0.0, 0.0, 1.0)  # Dir Names

    def change_text(self, path):
        self.load_location = str(path)


class SaveLoc(Screen):
    save_location = StringProperty('')
    data = ObjectProperty(None)

    def __init__(self, **args):
        Clock.schedule_once(self.init_widget, 0)
        return super(SaveLoc, self).__init__(**args)

    def init_widget(self, *args):
        fc = self.ids['filechooser_save']
        fc.bind(on_entry_added=self.update_file_list_entry)
        fc.bind(on_subentry_to_entry=self.update_file_list_entry)

    def update_file_list_entry(self, file_chooser, file_list_entry, *args):
        file_list_entry.children[0].color = (0.0, 0.0, 0.0, 1.0)  # File Names
        file_list_entry.children[1].color = (0.0, 0.0, 0.0, 1.0)  # Dir Names

    def change_text(self, path):
        self.save_location = str(path)


class Main(Screen):
    # def __init__(self):
    #     self.reset()
    #
    # def reset(self):
    def __init__(self, **kwargs):
        super(Main, self).__init__(**kwargs)
        # self.update_bar_trigger = Clock.create_trigger(self.update_bar, -1)

    location = StringProperty('')
    save_location = StringProperty('')
    experiment_id = ''  # identify by directory containing scores
    matrix_form = False

    input_test = ObjectProperty(None)
    test_label = ObjectProperty(None)
    train_perc = StringProperty('% Testing - 80 % Training')

    test_perc = NumericProperty(20)
    split = False
    normalize = ObjectProperty()
    norm_params = []

    detected_lbl = ObjectProperty('')
    modalities_lbl = ObjectProperty('')

    returned_modalities = StringProperty('')
    modalities = defaultdict(dict)
    modalities_original = defaultdict(dict)
    num_mods = NumericProperty(0)
    num_rocs = NumericProperty(0)

    # images
    display_path_density = StringProperty(None)
    dens_set = ObjectProperty()
    set_type = 'Entire'
    d_slide = Slider()

    r_slide = Slider()
    display_path_roc = StringProperty('')

    # roc_set = ObjectProperty()
    # roc_index = NumericProperty(0)
    # current_roc_nums = 0
    # roc_modality_pointer = 0

    # popups
    tanh_popup = ObjectProperty(Popup)
    dsig_popup = ObjectProperty(Popup)
    fusion_selection_popup = ObjectProperty(Popup)
    loading_pb = ProgressBar()
    increase_amount = NumericProperty(0)

    # messages
    msg_impgen_test = StringProperty('Imposter Samples: {} \n Genuine Samples: {}'.format(0, 0))
    msg_impgen_train = StringProperty('Imposter Samples: {} \n Genuine Samples: {}'.format(0, 0))

    msg_test = StringProperty('[b]TESTING: [/b] {} Subjects'.format(0))
    msg_train = StringProperty('[b]TRAINING: [/b] {} Subjects'.format(0))
    msg_modalities = StringProperty('[b]MODALITIES DETECTED: [/b] {}'.format(0))

    msg_accuracy = StringProperty('')
    msg_eer = StringProperty('')
    msg_fixed_tmr = StringProperty('')
    eval = defaultdict(lambda: defaultdict(partial(np.ndarray, 0)))
    fixed_FMR_val = TextInput()

    data_object = None

    def setup(self, path):
        self.load_path = path
        self.data_object = fm2.Score_data(path=self.load_path, test_perc=self.test_perc,
                             normalize=self.normalize, norm_params=self.norm_params,
                             lbl=self.ids.modalities_lbl)
        self.data_object.load_data()
        self.data_object.normalize_data()
        self.data_object.plot_distributions()

        self.densities = DensityPlots(slider=self.d_slide)
        self.densities.build_plot_list()
        self.display_path_density = self.densities.get_image_path()



        self.returned_modalities = ''.join([x + '\n' for x in self.data_object.get_modalities()])
        self.detected_lbl.opacity = 1.0
        # self.roc_index = 0

        self.beans = self.data_object.get_beans()
        num_gen_train = self.beans['gen_train']
        num_gen_test = self.beans['gen_test']

        num_imp_train = self.beans['imp_train']
        num_imp_test = self.beans['imp_test']

        # messages
        self.msg_impgen_test = '[b]Imposter Samples:[/b] {}\n[b]Genuine Samples:[/b] {}'.format(num_gen_test, num_imp_test)
        self.msg_impgen_train = '[b]Imposter Samples:[/b] {}\n[b]Genuine Samples:[/b] {}'.format(num_gen_train, num_imp_train)
        self.msg_test = '[b]TESTING: [/b] {} Subjects'.format(num_imp_test)
        self.msg_train = '[b]TRAINING: [/b] {} Subjects'.format(num_imp_train)
        self.num_mods = len(self.modalities.keys())
        self.msg_modalities = '[b]MODALITIES DETECTED: [/b] {}'.format(self.num_mods)

    def modality_update_helper(self, args):
        user_vals = self.edit_mods.get_updates()
        self.data_object.update_datas(user_vals)
        self.update_modality_label()

    def update_modality_label(self):
        self.ids.modalities_lbl.text = ''
        for mod_key in self.data_object.get_modalities():
            self.ids.modalities_lbl.text = self.ids.modalities_lbl.text + mod_key + '\n\n'

#############################################################
#############################################################
    def save_popup(self):
        self.save_settings = SavePop.SavePopup()
        popup = Popup(title="Save ", content=self.save_settings, size_hint=(None, None),
                            size=(600, 600))
        self.save_settings.set_pop(popup)
        popup.open()

    def modality_edit_popup(self):
        self.edit_mods = PopupModalityEdit.ModeEditPopup(modality_list=self.data_object.get_modalities())
        self.popup_popup = Popup(title="Edit Modalities", content=self.edit_mods, size_hint=(None, None),
                                size=(600, 600))
        self.edit_mods.set_pop(self.popup_popup)
        self.popup_popup.bind(on_dismiss=self.modality_update_helper)

        self.popup_popup.open()

    def reset_popup(self):
        self.reset = ResetPopup.ResetPopup()
        self.reset_popup = Popup(title="RESET WARNING ", content=self.reset, size_hint=(None, None),
                                size=(400, 400))
        self.reset.set_pop(self.reset_popup)
        self.reset_popup.open()  # show the popup

        #TODO
        if self.reset.check_reset():
            self.modalities = None
            self.location = ''
            self.ids.modalities_lbl.text = ''

    def Tanh_popup(self):
        self.show_tanh = TanhPopup.TanhPopup()
        self.tanh_popup = Popup(title="Tanh Estimator ", content=self.show_tanh, size_hint=(None, None),
                                size=(400, 400))
        self.tanh_popup.open()  # show the popup
        self.show_tanh.set_pop(self.tanh_popup)

    def DSig_popup(self):
        self.show_dsig = DSigPopup.DSigPopup()
        self.dsig_popup = Popup(title="Double Sigmoid Estimator ", content=self.show_dsig, size_hint=(None, None),
                                size=(400, 400))
        self.dsig_popup.open()  # show the popup
        self.show_dsig.set_pop(self.dsig_popup)

    def fusion_selective_popup(self):
        self.show_fusion_selection = SelectiveFusionPopup.SelectiveFusionPopup()
        self.fusion_selection_popup = Popup(title="Selective Fusion", content=self.show_fusion_selection, size_hint=(None, None),
                                size=(600, 600))
        self.show_fusion_selection.set_pop(self.fusion_selection_popup, self.data_object.get_modalities())

        self.fusion_selection_popup.open()  # show the popup

    #############################################################
    #############################################################

    def set_normalization(self):
        if not self.data_object:
            pass

        if self.chk_MinMax.active:
            self.normalize = 'MinMax'

        elif self.chk_ZScore.active:
            self.normalize = 'ZScore'

        elif self.chk_Decimal.active:
            self.normalize = 'Decimal'

        elif self.chk_Median.active:
            self.normalize = 'Median'

        elif self.chk_DSigmoid.active:
            self.normalize = 'DSigmoid'
            self.norm_params = self.show_dsig.get_dsig()

        elif self.chk_TanhEstimator.active:
            self.normalize = 'TanhEstimator'
            self.norm_params = self.show_tanh.get_tanh()

        elif self.chk_Biweight.active:
            self.normalize = 'BiweightEstimator'

        elif self.chk_ZScore.active:
            self.normalize = 'Zscore'

        else:
            self.normalize = 'None'

    def set_density_set(self):
        if self.chk_Test.active:
            self.display_path_density = self.densities.update_test_train('Testing')
        elif self.chk_Train.active:
            self.display_path_density = self.densities.update_test_train('Training')
        else:
            self.display_path_density = self.densities.update_test_train('Entire')

    def update_density_type(self):
        self.display_path_density = self.densities.update_plot_type()


    def next_roc_plot(self):
        self.display_path_roc = self.roc_object.update_plot()
    # def set_roc_set(self):
    #     files = []
    #     for filename in glob.glob('./generated/ROC/*'):
    #         files.append(filename)
    #
    #     self.roc_set = files
    #     self.r_slide.max = len(files)-1
    #     self.current_roc_nums = len(files)

    def density_slider(self, value):
        if 0 <= value < self.d_slide.max+1:
            self.display_path_density = self.densities.update_plot(value)

    def roc_slider(self, value):
        mx = self.r_slide.max
        if 0 <= value < self.r_slide.max-1:
            self.display_path_roc = self.roc_object.slider_update(value)


    def slider_button(self, direction, img_set, value):
        if img_set == 'density':
            if direction == 'left':
                self.display_path_density = self.densities.move_left()

            if direction == 'right':
                    self.display_path_density = self.densities.move_right()

        if img_set == 'roc':
            if direction == 'left':
                if 0 <= value:
                    self.display_path_roc = self.roc_object.move_left()

            if direction == 'right':
                if 0 <= value < self.r_slide.max:
                    self.display_path_roc = self.roc_object.move_right()


    def checkbox_click(self, instance, value):
        if value is True:
            self.split = True
            self.input_test.opacity = 1.0
            self.test_label.opacity = 1.0

        else:
            self.split = False
            self.input_test.opacity = 0.0
            self.test_label.opacity = 0.0


    def selective_fusion_click(self, instance, value):
        if value is True:
            self.fusion_selective_popup()


    def update_test(self, test_p):
        self.test_perc = test_p
        self.train_perc = '% Testing - ' + str(100 - self.test_perc) + '% Training'

    #########################################################################
    def save_report(self, save_location):
        if not os.path.exists(save_location + '/FusionReport/'):
            os.makedirs(save_location + '/FusionReport/')

        things_to_save = self.save_settings.get_save_reports()
        if 'report' in things_to_save:
            generate_summary(modalities=self.data_object.get_modalities(), results=self.eval,
                             roc_plt=self.display_path_roc,
                             fmr_rate=float(self.fixed_FMR_val.text),
                             save_to_path=save_location+ '/FusionReport/')

        if 'estimates' in things_to_save:
            if not os.path.exists(save_location + '/FusionReport/DensityEstimates/'):
                os.makedirs(save_location + '/FusionReport/DensityEstimates/')

            for filename in glob.glob('./generated/density/*'):
                shutil.copy(filename, save_location + '/FusionReport/DensityEstimates/')

        if 'rocs' in things_to_save:
            if not os.path.exists(save_location + '/FusionReport/ROCplots/'):
                os.makedirs(save_location + '/FusionReport/ROCplots/')

            for filename in glob.glob('./generated/ROC/*'):
                shutil.copy(filename, save_location + '/FusionReport/ROCplots/')

        if 'testtrain' in things_to_save:
            if not os.path.exists(save_location + '/FusionReport/TrainTestData/'):
                os.makedirs(save_location + '/FusionReport/TrainTestData/')

            for key, items in self.modalities.items():
                tmp_train = items['train_x'].transpose()
                tmp_test = items['test_x'].transpose()

                tmp_train_y = items['train_y'].transpose()
                tmp_test_y = items['test_y'].transpose()

                train = pd.DataFrame(data={'Data': tmp_train, 'Label': tmp_train_y})
                test = pd.DataFrame(data={'Data': tmp_test, 'Label': tmp_test_y})

                train.to_csv(save_location + '/FusionReport/TrainTestData/' + key + '-TRAIN.csv')
                test.to_csv(save_location + '/FusionReport/TrainTestData/' + key + '-TEST.csv')


        if 'datamets' in things_to_save:
            dataset_metrics = pd.DataFrame(data={'Modalities': self.data_object.get_modalities(),
                                                 'Train_Split': str(100-self.test_perc)+'%',
                                                 'Test_Split': str(self.test_perc)+'%',
                                                 'Total_Training': self.num_gen_train+self.num_imp_train,
                                                 'Total_Testing': self.num_gen_test+self.num_imp_test,
                                                 'Train_Genuine': self.num_gen_train,
                                                 'Train_Imposter': self.num_imp_train,
                                                 'Test_Genuine': self.num_gen_test,
                                                 'Test_Imposter': self.num_imp_test,
                                                 })

            dataset_metrics.to_csv(save_location + '/FusionReport/Dataset_Metrics.csv', index=False)

        if 'analysis' in things_to_save:
            eval_metrics = pd.DataFrame(self.eval)
            eval_metrics.to_csv(save_location+'/FusionReport/EvaluationMetrics.csv', index=False)

    def get_tpr(fpr, tpr, fixed_far=0.01):
        vert_line = np.full(len(fpr), fixed_far)
        idx = np.argwhere(np.diff(np.sign(fpr - vert_line))).flatten()
        return tpr[idx][0]

    def fuse(self):
        self.msg_accuracy = ''
        self.msg_eer = ''
        self.msg_fixed_tmr = ''

        fusion_list = []
        self.sequential_fusion_settings = None
        if self.chk_selective.active:
            fusion_list.append('SequentialRule')
            self.sequential_fusion_settings = self.show_fusion_selection.get_parms()
        if self.chk_sum.active:
            fusion_list.append('SumRule')
        if self.chk_svm.active:
            fusion_list.append('SVMRule')

        fusion_mod = Fuse.FuseRule(list_o_rules=fusion_list, score_data=self.data_object.score_data,
                                   modalities=self.data_object.get_modalities(),  fusion_settings=self.sequential_fusion_settings)

        mets = fusion_mod.fuse_all()
        fusion_mod.cmc()
        self.roc_object = ROCsPlots(slider=self.r_slide)
        self.display_path_roc = self.roc_object.build_plot_list()

        # build strings
        for fused in [x for x in mets.index if ':' in x]:
            accuracy = '[b]'+fused+': [/b] {}'.format(0) + self.truncate(mets.loc[fused]['AUC'], 6) + '\n'
            eer = '[b]'+fused+': [/b] {}'.format(0) + self.truncate(mets.loc[fused]['EER'], 6) + '\n'
            tmr = '[b]'+fused+': [/b] {}'.format(0) + self.truncate(self.get_TMR(tpr=mets.loc[fused]['TPRS'],
                                                                                 fpr=mets.loc[fused]['FPRS'],
                                                                                 fixed_far=float(self.fixed_FMR_val.text)), 6) + '\n'

            self.msg_accuracy = self.msg_accuracy + accuracy
            self.msg_eer = self.msg_eer + eer
            self.msg_fixed_tmr = self.msg_fixed_tmr + tmr

        # generate_summary(modalities=self.data_object.get_modalities(), results=self.eval,
        #                  roc_plt=self.display_path_roc,
        #                  fmr_rate=float(self.fixed_FMR_val.text))

    def update_evals(self):
        ## A Fixed FMR has been updated
        self.msg_fixed_tmr = ''
        for key, mods in self.eval.items():
            if 'Rule' in key:
                estimated_tmr = self.get_TMR(fpr=self.eval[key]['fprs'], tpr=self.eval[key]['tprs'],
                                             fixed_far=float(self.fixed_FMR_val.text))
                self.eval[key]['TMR'] = estimated_tmr
                tmr = '[b]' + key + ': [/b] {}'.format(0) + self.truncate(estimated_tmr, 6) + '\n'

                self.msg_fixed_tmr = self.msg_fixed_tmr + tmr

    def truncate(self, f, n):
        '''Truncates/pads a float f to n decimal places without rounding'''
        s = '{}'.format(f)
        if 'e' in s or 'E' in s:
            return '{0:.{1}f}'.format(f, n)
        i, p, d = s.partition('.')
        return '.'.join([i, (d + '0' * n)[:n]])

    def get_TMR(self, fpr, tpr, fixed_far):
        vert_line = np.full(len(fpr), fixed_far)
        idx = np.argwhere(np.diff(np.sign(fpr - vert_line))).flatten()
        return tpr[idx]


# class E(ExceptionHandler):
#     def handle_exception(self, inst):
#         return ExceptionManager.PASS

# ExceptionManager.add_handler(E())

presentation = Builder.load_file("styles.kv")


class TabbedPanelApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)

        self.title = 'Score Fusion'
        return presentation


if __name__ == '__main__':
    TabbedPanelApp().run()
