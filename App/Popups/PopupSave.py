import kivy
from kivy.uix.gridlayout import GridLayout
kivy.require('1.9.0')


class SavePopup(GridLayout):
    fpop = None

    def get_save_reports(self):
        things_to_save = []

        if self.chk_save_report.active:
            things_to_save.append('report')

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