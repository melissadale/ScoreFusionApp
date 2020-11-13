import kivy
from kivy.uix.gridlayout import GridLayout
kivy.require('1.9.0')

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
