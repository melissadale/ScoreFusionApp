import kivy
from kivy.uix.gridlayout import GridLayout
import shutil

kivy.require('1.9.0')


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
