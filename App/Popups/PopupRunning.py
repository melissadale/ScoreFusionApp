import kivy
from kivy.uix.gridlayout import GridLayout
import shutil

kivy.require('1.9.0')


class RunningPopup(GridLayout):
    fpop = None

    def check_reset(self):
        return self.really_reset

    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()
