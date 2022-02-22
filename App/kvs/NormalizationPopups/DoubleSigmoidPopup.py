import kivy
from kivy.uix.gridlayout import GridLayout
kivy.require('1.9.0')

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

    def get_params(self):
        return {'t': int(self.t), 'r1': int(self.r1), 'r2': int(self.r2)}