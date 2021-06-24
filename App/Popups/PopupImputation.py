import kivy
from kivy.uix.gridlayout import GridLayout
kivy.require('1.11.1')

class Imputation(GridLayout):
    dpop = None
    imputation = "ignore"

    def set_pop(self, pwin):
        self.dpop = pwin

    def close(self):
        self.dpop.dismiss()

    def set_imputation(self):
        if self.ids['chk_mean'].active:
            self.imp = ['Mean']

        elif self.ids['chk_median'].active:
            self.imp = ['Median']

        elif self.ids['chk_Bayesian'].active:
            self.imp = ['Bayesain']

        elif self.ids['chk_DT'].active:
            self.imp = ['DT']

        self.close()

    def get_imputation(self):
        return self.imputation
