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
            self.imputation = 'Mean'

        elif self.ids['chk_median'].active:
            self.imputation = 'Median'

        elif self.ids['chk_Bayesian'].active:
            self.imputation = 'Bayesian'

        elif self.ids['chk_DT'].active:
            self.imputation ='DT'

        elif self.ids['chk_KNN'].active:
            self.imputation = ['KNN', int(self.ids.k.text)]

        self.close()

    def get_imputation(self):
        return self.imputation

