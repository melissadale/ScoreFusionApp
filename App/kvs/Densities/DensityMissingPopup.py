import kivy
import pandas as pd
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
kivy.require('1.11.1')


class DensityMissingPopup(ScrollView):
    def __init__(self, **kwargs):
        super().__init__()
        self.modality_list = kwargs.get('modality_list')
        self.missing = kwargs.get('sparcity')
        if not self.modality_list:
            pass
        else:
            num_mods = len(self.modality_list)
            inc = num_mods

            for mod in self.modality_list:
                row = GridLayout(cols=7)

                ModName = Label(text='[color=F08521]' + mod + '[/color]', size_hint_x=0.2, size_hint_y=None,
                    markup = True)
                row.add_widget(ModName)

                bars = BoxLayout(orientation='horizontal')
                full = Label(text='[color=FFFFFF]' + str(round(self.missing.at[mod, '% Full']*100, 2)) + '%[/color]', size_hint_x=0.1, size_hint_y=None,
                    markup = True)
                row.add_widget(Widget(size_hint_x=0.2, size_hint_y=None))

                perc = Label(text='[color=FFFFFF]' + '' + '%[/color]', size_hint_x=0.1, size_hint_y=None,
                    markup = True)
                row.add_widget(full)

                # Used for FORCED spacing and alignment
                ## MID
                row.add_widget(Widget(size_hint_x=0.1, size_hint_y=None))
                ## MID

                MissingName = Label(text='[color=F08521]' + str(num_mods-inc) + ' Missing Modalities' + '[/color]', size_hint_x=0.2, size_hint_y=None,
                    markup = True)
                row.add_widget(MissingName)
                inc = inc - 1

                row.add_widget(Widget(size_hint_x=0.2, size_hint_y=None))

                perc = Label(text='[color=FFFFFF]' + '%' + '[/color]', size_hint_x=0.1, size_hint_y=None,
                    markup = True)
                row.add_widget(perc)

                self.ids['modalities'].add_widget(row)


    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()