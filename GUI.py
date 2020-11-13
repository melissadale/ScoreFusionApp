"""
Created on 1/29/2020
By Melissa Dale
"""
import functools
import glob
import os
import pickle
import shutil

# os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'
import threading

import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial
from numpy import ones, vstack
from numpy.linalg import lstsq

import Analytics.format_data as fm
import Analytics.Fuse as Fuse
import AppWidgets.PopupSave as SavePop
import AppWidgets.PopupReset as ResetPopup
import AppWidgets.PopupTanh as TanhPopup
import AppWidgets.PopupDSig as DSigPopup
import AppWidgets.PopupSelectiveFusion as SelectiveFusionPopup

from AppWidgets.ReportPDFs import generate_summary

# Program to explain how to create tabbed panel App in kivy: https://www.geeksforgeeks.org/python-tabbed-panel-in-kivy/
import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.base import ExceptionManager, ExceptionHandler
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
    detected_lbl = ObjectProperty()
    train_perc = StringProperty('% Testing - 80 % Training')

    test_perc = NumericProperty(20)
    split = False
    normalize = ObjectProperty()
    norm_params = []

    detected_lbl = ObjectProperty('')
    modalities_lbl = ObjectProperty('')

    returned_modalities = StringProperty('')
    modalities = defaultdict(dict)
    num_mods = NumericProperty(0)
    num_rocs = NumericProperty(0)

    # images
    display_path_density = StringProperty(None)
    dens_set = ObjectProperty()
    set_type = 'Entire'
    d_slide = Slider(min=0, max=1, value=0)
    r_slide = Slider(min=0, max=1, value=0)

    display_path_roc = StringProperty(None)
    roc_set = ObjectProperty()
    roc_index = NumericProperty(0)
    current_roc_nums = 0

    density_type = [ 'hist','PDF', 'overlap']
    density_type_pointer = 2
    current_density = density_type[density_type_pointer]
    density_modality_pointer = 0
    roc_modality_pointer = 0
    #popups
    save_popup = ObjectProperty(Popup)
    reset_popup = ObjectProperty(Popup)
    tanh_popup = ObjectProperty(Popup)
    dsig_popup = ObjectProperty(Popup)
    fusion_selection_popup = ObjectProperty(Popup)
    modality_list = ObjectProperty(None)

    # Progress bars
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

    ### Progress Bar Things
    def update_bar(self, mod_key, dt):
        if self.loading_pb.value <= 100:
            self.loading_pb.value += self.increase_amount
            self.ids.modalities_lbl.text = self.ids.modalities_lbl.text + mod_key + '\n\n'

    def setup_2(self, path):
        self.load_path = path
        threading.Thread(target=self.setup, args=()).start()

    def setup(self):
        temp_modalities, self.matrix_form, e_id = fm.get_data(self.load_path, test_perc=self.test_perc, dissimilar=self.chk_dissimilar)

        self.increase_amount = 100/len(temp_modalities)

        for key, items in temp_modalities.items():
            mod_dict = fm.split_data(items, normalize=self.normalize, norm_params=self.norm_params, key=key, exp_id=e_id)
            self.modalities[key] = mod_dict
            Clock.schedule_once(functools.partial(self.update_bar, key))
        self.modality_list = list(self.modalities)

        self.detected_lbl.opacity = 1.0
        self.roc_index = 0

        self.num_gen_train = np.count_nonzero(self.modalities[list(self.modalities)[0]]['train_y'] == 0)
        self.num_gen_test = np.count_nonzero(self.modalities[list(self.modalities)[0]]['test_y'] == 0)

        self.num_imp_train = np.count_nonzero(self.modalities[list(self.modalities)[0]]['train_y'] == 1)
        self.num_imp_test = np.count_nonzero(self.modalities[list(self.modalities)[0]]['test_y'] == 1)

        # messages
        self.msg_impgen_test ='[b]Imposter Samples:[/b] {}\n[b]Genuine Samples:[/b] {}'.format(self.num_gen_test, self.num_imp_test)
        self.msg_impgen_train = '[b]Imposter Samples:[/b] {}\n[b]Genuine Samples:[/b] {}'.format(self.num_gen_train, self.num_imp_train)
        self.msg_test = '[b]TESTING: [/b] {} Subjects'.format(self.num_imp_test)
        self.msg_train = '[b]TRAINING: [/b] {} Subjects'.format(self.num_imp_train)
        self.num_mods = len(self.modalities.keys())
        self.msg_modalities = '[b]MODALITIES DETECTED: [/b] {}'.format(self.num_mods)

        self.dens_set = self.get_right_densityplots()
        self.display_path_density = self.dens_set[self.density_modality_pointer]
        self.d_slide.max = len(self.modality_list)-1

#############################################################
#############################################################
    def save_popup(self):
        self.save_settings = SavePop.SavePopup()
        self.popup_popup = Popup(title="Save ", content=self.save_settings, size_hint=(None, None),
                                size=(600, 600))
        self.save_settings.set_pop(self.popup_popup)
        self.popup_popup.open()  # show the popup

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
        self.show_fusion_selection.set_pop(self.fusion_selection_popup, self.modality_list)

        self.fusion_selection_popup.open()  # show the popup

    #############################################################
    #############################################################

    def set_normalization(self):
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
            self.set_type = 'Testing'
            self.dens_set = self.get_right_densityplots()
            self.display_path_density = self.dens_set[self.density_modality_pointer]

        elif self.chk_Train.active:
            self.set_type = 'Training'
            self.dens_set = self.get_right_densityplots()
            self.display_path_density = self.dens_set[self.density_modality_pointer]
        else:
            self.set_type = 'Entire'
            self.dens_set = self.get_right_densityplots()
            self.display_path_density = self.dens_set[self.density_modality_pointer]

    def set_roc_set(self):
        files = []
        for filename in glob.glob('./generated/ROC/*'):
            files.append(filename)

        self.roc_set = files
        self.r_slide.max = len(files)-1
        self.current_roc_nums = len(files)

    def get_right_densityplots(self):
        files = []
        pointer = './generated/density/' + self.set_type+'/' + self.current_density

        for filename in glob.glob(pointer + '/*'):
            if self.set_type in filename and self.experiment_id in filename:
                files.append(filename)

        return files

    def update_density_type(self):
        self.density_type_pointer = (self.density_type_pointer + 1) % 3
        self.current_density = self.density_type[self.density_type_pointer]
        self.set_density_set()

    def density_slider(self, value):
        if 0 <= value < self.d_slide.max+1:
            self.display_path_density = self.dens_set[value]

    def roc_slider(self, value):
        self.set_roc_set()

        if 0 <= value < self.r_slide.max+1:
            self.display_path_roc = self.roc_set[value]

    def slider_button(self, direction, img_set, value):
        if img_set == 'density':
            if direction == 'left':
                if self.density_modality_pointer > 0:
                    self.display_path_density = self.dens_set[value - 1]
                    self.density_modality_pointer = self.density_modality_pointer - 1

            if direction == 'right':
                if self.density_modality_pointer < self.d_slide.max:
                    self.display_path_density = self.dens_set[value + 1]
                    self.density_modality_pointer = self.density_modality_pointer + 1

        if img_set == 'roc':
            if direction == 'left':
                if self.roc_modality_pointer > 0:
                    self.display_path_roc = self.roc_set[value - 1]
                    self.roc_modality_pointer = self.roc_modality_pointer - 1

            if direction == 'right':
                if self.roc_modality_pointer < self.r_slide.max:
                    self.display_path_roc = self.roc_set[value + 1]
                    self.roc_modality_pointer = self.roc_modality_pointer + 1


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
            generate_summary(modalities=self.modality_list, results=self.eval,
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
            dataset_metrics = pd.DataFrame(data={'Modalities': list(self.modalities),
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

    def fuse(self):
        self.msg_accuracy = ''
        self.msg_eer = ''
        self.msg_fixed_tmr = ''

        fusion_list = []
        self.serial_fusion_settings = None
        if self.chk_selective.active:
            fusion_list.append('SerialRule')
            self.serial_fusion_settings = self.show_fusion_selection.get_parms()
        if self.chk_sum.active:
            fusion_list.append('SumRule')
        if self.chk_svm.active:
            fusion_list.append('SVMRule')

        fusion_mod = Fuse.FuseRule(fusion_list, self.modalities, self.normalize, self.serial_fusion_settings, self.matrix_form)

        mets, disp_pth = fusion_mod.fuse_all()
        self.set_roc_set()

        self.display_path_roc = disp_pth[0]

        # build strings
        for key, mods in mets.items():
            if 'Rule' in key:
                self.eval[key]['AUC'] = mods['AUC']
                self.eval[key]['EER'] = mods['EER']

                self.eval[key]['fprs'] = mods['fprs']
                self.eval[key]['tprs'] = mods['tprs']
                estimated_tmr = self.get_TMR(mods['fprs'], mods['tprs'],
                                                     float(self.fixed_FMR_val.text))
                self.eval[key]['TMR'] = estimated_tmr

        for key, vals in self.eval.items():
                accuracy = '[b]'+key+': [/b] {}'.format(0) + self.truncate(vals['AUC'], 6) + '\n'
                eer = '[b]'+key+': [/b] {}'.format(0) + self.truncate(vals['EER'], 6) + '\n'
                tmr = '[b]'+key+': [/b] {}'.format(0) + self.truncate(vals['TMR'], 6) + '\n'

                self.msg_accuracy = self.msg_accuracy + accuracy
                self.msg_eer = self.msg_eer + eer
                self.msg_fixed_tmr = self.msg_fixed_tmr + tmr

        generate_summary(modalities=self.modality_list, results=self.eval,
                         roc_plt=self.display_path_roc,
                         fmr_rate=float(self.fixed_FMR_val.text))

    def update_evals(self):
        self.msg_fixed_tmr = ''
        for key, mods in self.eval.items():
            if 'Rule' in key:
                estimated_tmr = self.get_TMR(self.eval[key]['fprs'], self.eval[key]['tprs'],
                                             float(self.fixed_FMR_val.text))
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

    def get_TMR(self, fmr, tmr, fixed_FMR):
        df = pd.DataFrame({'FMR': fmr, 'TMR': tmr})
        df = df.sort_values('FMR')

        for idx, row in df.iterrows():
            if row['FMR'] < fixed_FMR:
                continue
            else:
                break

        if idx != 0:
            p1_FMR = df.iloc[idx - 1]['FMR']
            p1_TMR = df.iloc[idx - 1]['TMR']

            p2_FMR = df.iloc[idx]['FMR']
            p2_TMR = df.iloc[idx]['TMR']

        else:
            p1_FMR = df.iloc[idx]['FMR']
            p1_TMR = df.iloc[idx]['TMR']

            p2_FMR = df.iloc[idx+1]['FMR']
            p2_TMR = df.iloc[idx+1]['TMR']

        m, b = self.get_line([(p1_FMR, p1_TMR), (p2_FMR, p2_TMR)])
        estimated_tmr = m * fixed_FMR + b
        return estimated_tmr

    def get_line(self, points):
        """
        https://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
        """
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        return m, c

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
