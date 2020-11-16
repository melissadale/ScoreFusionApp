import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.label import Label
kivy.require('1.11.1')


class ModeEditPopup(GridLayout):
    def __init__(self, **kwargs):
        super().__init__()
        self.modality_list = kwargs.get('modality_list')
        self.cols = 1

        if not self.modality_list:
            pass
        else:

            header = GridLayout(cols=4, size_hint_y=0.1)
            header.add_widget(Label(text='Modality Name', size_hint_x=0.25, bold=True,
                                    color=(0.05, 0.69, 0.29, 1.0)))
            header.add_widget(Label(text='', size_hint_x=0.25))
            header.add_widget(Label(text='Similarity ', size_hint_x=0.25, bold=True))
            header.add_widget(Label(text='Dissimilarity ', size_hint_x=0.25, bold=True))
            self.add_widget(header)

            layout = GridLayout(cols=1, size_hint_y=0.8)
            for mod in self.modality_list:
                row = GridLayout(cols=4)
                row.add_widget(TextInput(text=mod, size_hint_x=0.5, size_hint_y=None, multiline=False,
                                         height=30, valign='top'))
                row.add_widget(CheckBox(size_hint_x=0.25, size_hint_y=None, active=True))
                row.add_widget(CheckBox(size_hint_x=0.25, size_hint_y=None))

                layout.add_widget(row)
            layout.add_widget(Label(text=''))

            self.add_widget(layout)
            self.add_widget(Button(text='Update', size_hint_y=0.1, bold=True,
                                   background_color=(0.05, 0.69, 0.29, 1.0)))

    fpop = None

    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()




