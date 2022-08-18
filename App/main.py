import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kvs.DataPanel import Data
from kvs.DensityPanel import Densities
from kvs.FusionPanel import FusionPanel
from kvs.ResultsPanel import ResultsPanel

kivy.require('2.0.0')
Builder.load_file('SFA.kv')


class ScreenManagement(ScreenManager):
    pass


class DirectoryScreen(Screen):
    def __init__(self, **kwargs):
        super(DirectoryScreen, self).__init__(**kwargs)
        self.init_widget()

    def init_widget(self, *args):
        fc = self.ids['filechooser']
        fc.bind(on_entry_added=self.update_file_list_entry)
        fc.bind(on_subentry_to_entry=self.update_file_list_entry)

    def update_file_list_entry(self, file_chooser, file_list_entry, *args):
        file_list_entry.children[0].color = (0.0, 0.0, 0.0, 1.0)  # File Names
        file_list_entry.children[1].color = (0.0, 0.0, 0.0, 1.0)  # Dir Names`


class PanelLayout(Screen):
    def __init__(self, **kwargs):
        super(PanelLayout, self).__init__(**kwargs)

        # add Tab Panels
        self.dat = Data()
        self.ids['data_input_panel'].add_widget(self.dat)

        self.density = Densities()
        self.ids['density_panel'].add_widget(self.density)

        self.fusion_tab = FusionPanel()
        self.ids['fusion_panel'].add_widget(self.fusion_tab)

        self.results_tab = ResultsPanel()
        self.ids['results_panel'].add_widget(self.results_tab)

    def data_path(self, pth):
        self.dat.set_data_location(pth)

    def update_beans(self):
        self.dat.get_data_files()
        self.density.set_beans(self.dat.beans, self.dat.sparcity, self.dat.score_data.get_modalities())


screen_manager = ScreenManager()
screen_manager.add_widget(PanelLayout(name="main_screen"))
screen_manager.add_widget(DirectoryScreen(name="directory_screen"))


class FusionApp(App):
    def build(self):
        self.title = 'Score Fusion'
        return screen_manager


if __name__ == '__main__':
    FusionApp().run()
