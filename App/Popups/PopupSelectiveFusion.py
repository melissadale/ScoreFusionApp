import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
kivy.require('1.9.0')

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
        return {'alpha': float(self.alpha), 'beta': float(self.beta), 'baseline': self.baseline,
                'auto': self.chk_auto_alpha}

