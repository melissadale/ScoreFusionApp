from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock


class Loc(Screen):
    load_location = StringProperty('')
    data = ObjectProperty(None)

    def __init__(self, **args):
        Clock.schedule_once(self.init_widget, 0)
        return super(Loc, self).__init__(**args)

    def init_widget(self, *args):
        fc = self.ids['filechoose']
        fc.bind(on_entry_added=self.update_file_list_entry)
        fc.bind(on_subentry_to_entry=self.update_file_list_entry)

    def update_file_list_entry(self, file_choose, file_list_entry, *args):
        file_list_entry.children[0].color = (0.0, 0.0, 0.0, 1.0)  # File Names
        file_list_entry.children[1].color = (0.0, 0.0, 0.0, 1.0)  # Dir Names

    def change_text(self, path):
        self.load_location = str(path)


class SaveLoc(Screen):
    save_location = StringProperty('')
    data = ObjectProperty(None)

    def __init__(self, **args):
        Clock.schedule_once(self.init_widget, 0)
        return super(SaveLoc, self).__init__(**args)

    def init_widget(self, *args):
        fc = self.ids['filechooser_save']
        fc.bind(on_entry_added=self.update_file_list_entry)
        fc.bind(on_subentry_to_entry=self.update_file_list_entry)

    def update_file_list_entry(self, file_chooser, file_list_entry, *args):
        file_list_entry.children[0].color = (0.0, 0.0, 0.0, 1.0)  # File Names
        file_list_entry.children[1].color = (0.0, 0.0, 0.0, 1.0)  # Dir Names

    def change_text(self, path):
        self.save_location = str(path)

    def get_save_location(self):
        return self.save_location