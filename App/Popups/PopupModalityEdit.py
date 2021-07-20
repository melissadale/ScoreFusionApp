import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget

kivy.require('1.11.1')


class ModeEditPopup(GridLayout):
    def __init__(self, **kwargs):
        super().__init__()
        self.done = False
        self.modality_list = kwargs.get('modality_list')
        self.msg = kwargs.get('msg')
        self.cols = 1
        self.return_vals = None
        self.fpop = None

        if not self.modality_list:
            pass
        else:
            self.changes = dict.fromkeys(set(self.modality_list), [])

            header = BoxLayout(cols=4, size_hint_y=0.1,)

            header.add_widget(Label(text='Fuse', size_hint_x=None, width=10, bold=True,
                                    text_size = self.size, halign="right", valign="middle"))
            header.add_widget(Label(text='Modality',size_hint_x=None, width=290, bold=True,
                                    text_size = self.size, color=(0.05, 0.69, 0.29, 1.0),
                                    halign="left", valign="middle"))
            header.add_widget(Label(text='Similarity ', size_hint_x=None, width=100, bold=True,
                                    text_size = self.size, halign="right", valign="middle"))
            header.add_widget(Label(text='Dissimilarity ', size_hint_x=None, width=100, bold=True,
                                    text_size = self.size, halign="right", valign="middle"))

            self.add_widget(header)

            layout = GridLayout(cols=1, size_hint_y=0.8)

            if '_ORIGINAl' in self.modality_list:
                original = [x for x in self.modality_list if "_ORIGINAL" in self.modality_list]
            else:
                original = self.modality_list

            for mod in original:
                row = BoxLayout()

                fuse_chk = CheckBox(width=30, height=30, size_hint_x=None, size_hint_y=None,
                                    active=True)
                row.add_widget(fuse_chk)
                row.add_widget(Widget(width=15, height=30, size_hint_x=None, size_hint_y=None))

                mod_name = TextInput(text=mod, width=250, height=30, size_hint_x=None, size_hint_y=None,
                                     multiline=False)
                row.add_widget(mod_name)
                row.add_widget(Widget(width=60, height=30, size_hint_x=None, size_hint_y=None))


                similarity_chk = CheckBox(width=30, height=30, size_hint_x=None, size_hint_y=None,
                                    active=True, group=mod)
                row.add_widget(similarity_chk)
                row.add_widget(Widget(width=60, height=30, size_hint_x=None, size_hint_y=None))

                dissimilarity_chk = CheckBox(width=50, height=30, size_hint_x=None, size_hint_y=None,
                                    group=mod)
                row.add_widget(dissimilarity_chk)

                layout.add_widget(row)
                self.changes[mod] = [mod_name, similarity_chk, dissimilarity_chk, fuse_chk]

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
        tmp_message = ''

        for mod, val in tmp.items():
            mod_lbl = self.changes[mod][0].text
            mod_sim = self.changes[mod][1].active
            mod_dis = self.changes[mod][2].active
            use_mod = self.changes[mod][3].active
            tmp[mod] = [mod_lbl, mod_sim, mod_dis, use_mod]

            if self.changes[mod][3].active:
                tmp_message = tmp_message + mod_lbl + '\n\n'

        self.msg.text = tmp_message
        self.return_vals = tmp
        self.close()

    def get_updates(self):
        return self.return_vals