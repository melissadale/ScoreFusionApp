import kivy
from kivy.uix.gridlayout import GridLayout
kivy.require('1.9.0')

class TanhPopup(GridLayout):
    c = 0

    fpop = None

    def set_tanh(self, c):
        self.c = c

    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()

    def get_params(self):
        return float(self.c)