import kivy
from kivy.uix.gridlayout import GridLayout
kivy.require('1.11.1')

class Imputation(GridLayout):
    dpop = None
    imputation = "ignore"


    def set_imputation(self, imp):
        self.imputation = imp

    def set_pop(self, pwin):
        self.dpop = pwin

    def close(self):
        self.dpop.dismiss()

    def get_imputation(self):
        return self.imputation