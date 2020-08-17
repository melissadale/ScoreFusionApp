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
kivy.require('1.9.0')

##############################################################################################
# https://techwithtim.net/tutorials/kivy-tutorial/popup-windows/
##############################################################################################
class SavePopup(GridLayout):
    fpop = None

    def get_save_reports(self):
        things_to_save = []

        if self.chk_save_estimates.active:
            things_to_save.append('estimates')

        if self.chk_save_ROC.active:
            things_to_save.append('rocs')

        if self.chk_save_testtrain.active:
            things_to_save.append('testtrain')

        if self.chk_save_datamets.active:
            things_to_save.append('datamets')

        if self.chk_save_analysis.active:
            things_to_save.append('analysis')

        # chk_save_project

        return things_to_save

    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()

class ResetPopup(GridLayout):
    fpop = None
    really_reset = False

    def reset(self):
        self.really_reset = True
        print('Delete everything in Generated')
        try:
            shutil.rmtree('./generated')
        except:
            pass

    def check_reset(self):
        return self.really_reset

    def set_pop(self, pwin):
        self.fpop = pwin

    def save_popup(self):
        save = SavePopup()
        save_pop = Popup(title="Save ", content=save, size_hint=(None, None),
                                size=(600, 600))

        save.set_pop(save_pop)
        save_pop.open()  # show the popup

    def close(self):
        self.fpop.dismiss()

class TanhPopup(GridLayout):
    a = 0
    b = 0
    c = 0

    fpop = None

    def set_tanh(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()

    def get_tanh(self):
        return [int(self.a), int(self.b), int(self.c)]


class DSigPopup(GridLayout):
    t = 0
    r1 = 0
    r2 = 0

    dpop = None

    def set_dsig(self, t, r1, r2):
        self.t = t
        self.r1 = r1
        self.r2 = r2

    def set_pop(self, pwin):
        self.dpop = pwin

    def close(self):
        self.dpop.dismiss()

    def get_dsig(self):
        return [int(self.t), int(self.r1), int(self.r2)]

#######################################################################################################
#######################################################################################################

class SelectiveFusionPopup(GridLayout):
    alpha = 0
    beta = 0
    baseline = 0
    auto_alpha = False
    see_alpha = ObjectProperty(None)

    fpop = None
    modality_list = []

    def set_parms(self, a, b, c):
        self.alpha = a
        self.beta = b
        self.baseline = c

    def set_pop(self, pwin, mod_list):
        self.fpop = pwin
        self.modality_list = mod_list

    def show_dropdown(self, button, *largs):
        dp = DropDown()
        dp.bind(on_select=lambda instance, x: setattr(button, 'text', x))
        for i in self.modality_list:
            item = Button(text=i, size_hint_y=None, height=44)
            item.bind(on_release=lambda btn: dp.select(btn.text))
            dp.add_widget(item)
        dp.open(button)


    def check_auto(self, instance, state):
        if state is True:
            self.chk_auto_alpha = True
            self.see_alpha.opacity = 0.0
        else:
            self.chk_auto_alpha = False
            self.see_alpha.opacity = 1.0

    def close(self):
        self.fpop.dismiss()

    def get_parms(self):
        print(self.chk_auto_alpha)
        return {'alpha': float(self.alpha), 'beta': float(self.beta), 'baseline': self.baseline, 'auto': self.chk_auto_alpha}

