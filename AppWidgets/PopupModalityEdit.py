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
        self.done = False
        self.modality_list = kwargs.get('modality_list')
        self.cols = 1
        self.return_vals = None
        self.fpop = None

        if not self.modality_list:
            pass
        else:
            self.changes = dict.fromkeys(set(self.modality_list), [])

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
                mod_name = TextInput(text=mod, size_hint_x=0.5, size_hint_y=None, multiline=False,
                                     height=30, valign='top')
                row.add_widget(mod_name)

                similarity_chk = CheckBox(size_hint_x=0.25, size_hint_y=None, active=True)
                row.add_widget(similarity_chk)

                dissimilarity_chk = CheckBox(size_hint_x=0.25, size_hint_y=None)
                row.add_widget(dissimilarity_chk)

                layout.add_widget(row)
                self.changes[mod] = [mod_name, similarity_chk, dissimilarity_chk]
            layout.add_widget(Label(text=''))

            self.add_widget(layout)
            update_btn = Button(text='Update', size_hint_y=0.1, bold=True,
                                background_color=(0.05, 0.69, 0.29, 1.0))
            update_btn.bind(on_press=self.update_dicts)
            self.add_widget(update_btn)

    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()
        self.done = True
        pass

    def update_dicts(self, arg):
        tmp = dict.fromkeys(set(self.modality_list), [])

        for mod, val in tmp.items():
            mod_lbl = self.changes[mod][0].text
            mod_sim = self.changes[mod][1].active
            mod_dis = self.changes[mod][2].active
            tmp[mod] = [mod_lbl, mod_sim, mod_dis]

        self.return_vals = tmp
        self.close()

    def get_updates(self):
        return self.return_vals