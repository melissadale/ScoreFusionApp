"""
Created on 1/29/2020
By Melissa Dale
"""
import glob
import os
import pickle
import shutil

# os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'

import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial
from skimage import io
import skimage
import Analytics.format_data as fm
import Analytics.Fuse as Fuse
import AppWidgets.Popups as Popups

# Program to explain how to create tabbed panel App in kivy: https://www.geeksforgeeks.org/python-tabbed-panel-in-kivy/
import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.base import ExceptionManager, ExceptionHandler
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from AppWidgets.MyProgressBar import MyProgressBar

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




#
# class CustomDropDown(DropDown):
#     pass

class Main(Screen):
    # def __init__(self):
    #     self.reset()
    #
    # def reset(self):
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
    modalities = None
    num_mods = NumericProperty(0)
    num_rocs = NumericProperty(0)

    # images
    display_path_density = StringProperty(None)
    dens_set = ObjectProperty()
    set_type = 'Entire'
    d_slide = Slider(min=0, max=1, value=0)

    display_path_roc = StringProperty(None)
    current_roc_nums = 0
    roc_set = ObjectProperty()
    roc_index = NumericProperty(0)

    density_type = [ 'hist','PDF', 'overlap']
    density_type_pointer = 2
    current_density = density_type[density_type_pointer]
    density_modality_pointer = 0

    #popups
    save_popup = ObjectProperty(Popup)
    show_save = None

    reset_popup = ObjectProperty(Popup)
    reset_save = None

    tanh_popup = ObjectProperty(Popup)
    show_tanh = None

    dsig_popup = ObjectProperty(Popup)
    show_dsig = None

    fusion_selection_popup = ObjectProperty(Popup)
    show_fusion_selection = None
    select_fusion_settings = None

    modality_list = ObjectProperty(None)


    # Progress bars
    loading_pb = MyProgressBar()

    # messages
    msg_impgen_test = StringProperty('Imposter Samples: {} \n Genuine Samples: {}'.format(0, 0))
    msg_impgen_train = StringProperty('Imposter Samples: {} \n Genuine Samples: {}'.format(0, 0))

    msg_test = StringProperty('[b]TESTING: [/b] {} Subjects'.format(0))
    msg_train = StringProperty('[b]TRAINING: [/b] {} Subjects'.format(0))
    msg_modalities = StringProperty('[b]MODALITIES DETECTED: [/b] {}'.format(0))

    msg_accuracy = StringProperty('')
    msg_eer = StringProperty('')
    msg_f1 = StringProperty('')
    eval = defaultdict(lambda: defaultdict(partial(np.ndarray, 0)))

    ##### dataset metrics
    num_gen_train = 0
    num_gen_test = 0
    num_imp_train = 0
    num_imp_test = 0


    def setup(self, path):
        print('setup data stuff')

        self.modalities, self.experiment_id, self.matrix_form = fm.get_data(path, test_perc=self.test_perc, normalize=self.normalize, norm_params=self.norm_params, progress_bar=self.loading_pb)
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

        tmp = ''
        for key, item in self.modalities.items():
            tmp = tmp + key + '\n\n'
        self.ids.modalities_lbl.text = tmp


        self.dens_set = self.get_right_densityplots()
        self.display_path_density = self.dens_set[self.density_modality_pointer]
        self.d_slide.max = len(self.modality_list)-1

#############################################################
#############################################################
    def save_popup(self):
        self.save_settings = Popups.SavePopup()
        self.popup_popup = Popup(title="Save ", content=self.save_settings, size_hint=(None, None),
                                size=(600, 600))
        self.save_settings.set_pop(self.popup_popup)
        self.popup_popup.open()  # show the popup

    def reset_popup(self):
        self.reset = Popups.ResetPopup()
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
        self.show_tanh = Popups.TanhPopup()
        self.tanh_popup = Popup(title="Tanh Estimator ", content=self.show_tanh, size_hint=(None, None),
                                size=(400, 400))
        self.tanh_popup.open()  # show the popup
        self.show_tanh.set_pop(self.tanh_popup)

    def DSig_popup(self):
        self.show_dsig = Popups.DSigPopup()
        self.dsig_popup = Popup(title="Double Sigmoid Estimator ", content=self.show_dsig, size_hint=(None, None),
                                size=(400, 400))
        self.dsig_popup.open()  # show the popup
        self.show_dsig.set_pop(self.dsig_popup)

    def fusion_selective_popup(self):
        self.show_fusion_selection = Popups.SelectiveFusionPopup()
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

    def set_roc_set(self): # TODO
        files = []
        for filename in glob.glob('./generated/ROC/*'):
            files.append(filename)

        self.dens_set = files
        self.current_roc_nums = len(files)


    def get_right_densityplots(self): # TODO

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
        try:
            self.display_path_roc = self.roc_set[value]
        except:
            print('whoops, slider did not work')


    def slider_button(self, direction, img_set, value):
        try:
            if img_set == 'density':
                if direction == 'left':
                    if self.density_modality_pointer > 0:
                        self.display_path_density = self.dens_set[value - 1]
                        self.density_modality_pointer = self.density_modality_pointer - 1

                if direction == 'right':
                    if self.density_modality_pointer < self.d_slide.max:
                        self.display_path_density = self.dens_set[value + 1]
                        self.density_modality_pointer = self.density_modality_pointer + 1

        except Exception as c:
            print(c)

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
        fusion_list = []
        if self.chk_selective.active:
            fusion_list.append('SelectiveRule')
            self.select_fusion_settings = self.show_fusion_selection.get_parms()
        if self.chk_sum.active:
            fusion_list.append('SumRule')
        if self.chk_svm.active:
            fusion_list.append('SVMRule')

        fusion_mod = Fuse.FuseRule(fusion_list, self.modalities, self.normalize, self.select_fusion_settings, self.matrix_form)
        mets, disp_pth = fusion_mod.fuse_all()
        self.set_roc_set()

        self.display_path_roc = disp_pth[0]

        # build strings
        for key, mods in mets.items():
            if 'Rule' in key:
                self.eval[key]['AUC'] = mets[key]['AUC']
                self.eval[key]['EER'] = mets[key]['EER']
                self.eval[key]['F1'] = mets[key]['F1']

                accuracy = '[b]'+key+': [/b] {}'.format(0) + str(format(mets[key]['AUC'], '1.4f')) + '\n'
                eer = '[b]'+key+': [/b] {}'.format(0) + str(format(mets[key]['EER'], '1.4f')) + '\n'
                f1 = '[b]'+key+': [/b] {}'.format(0) + str(format(mets[key]['F1'], '1.4f')) + '\n'

                self.msg_accuracy = self.msg_accuracy + accuracy
                self.msg_eer = self.msg_eer + eer
                self.msg_f1 = self.msg_f1 + f1


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
