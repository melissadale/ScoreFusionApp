import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.widget import Widget
from kivy.uix.label import Label

kivy.require('1.11.1')


class ModeEditPopup(GridLayout):
    def __init__(self, **kwargs):
        super().__init__()
        self.modality_list = kwargs.get('modality_list')
        self.fpop = None
        self.changes = {}
        self.return_values = None

        if not self.modality_list:
            pass
        else:
            for mod in self.modality_list:
                row = BoxLayout(orientation="horizontal")

                fuse_chk = CheckBox(size_hint_x=0.2, size_hint_y=None, height=30,
                                    active=True)
                row.add_widget(fuse_chk)
                # Used for FORCED spacing and alignment
                row.add_widget(Widget(width=15, height=30, size_hint_x=None, size_hint_y=None))

                mod_name = TextInput(size_hint_x=0.4, size_hint_y=None, height=30,
                                     multiline=False, text=mod)
                row.add_widget(mod_name)
                row.add_widget(Widget(width=60, height=30, size_hint_x=None, size_hint_y=None))

                similarity_chk = CheckBox(size_hint_x=0.2, size_hint_y=None,
                                          active=True, group=mod, height=30)
                row.add_widget(similarity_chk)
                row.add_widget(Widget(width=60, height=30, size_hint_x=None, size_hint_y=None))

                dissimilarity_chk = CheckBox(size_hint_x=0.2, size_hint_y=None,
                                             group=mod, height=30)
                row.add_widget(dissimilarity_chk)

                self.ids['modalities'].add_widget(row)
                self.changes[mod] = [mod_name, similarity_chk, dissimilarity_chk, fuse_chk]

        self.ids['modalities'].add_widget(Label(text=''))

    def set_pop(self, pwin):
        self.fpop = pwin

    def close(self):
        self.fpop.dismiss()

    def update_dicts(self):
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
        tmp['new_msg'] = tmp_message
        self.return_values = tmp
        self.close()

    def get_updates(self):
        return self.return_values
