import kivy
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.clock import Clock
import functools

# Custom Imports
from kvs.ImputationPopup import Imputation as ImputationPop
from kvs.NormalizationPopups.DoubleSigmoidPopup import DSigPopup
from kvs.NormalizationPopups.TanhPopup import TanhPopup
from Objects.ScoreData import ScoreData

# styles
kivy.require('2.0.0')
Builder.load_file('kvs/density.kv')



class Densities(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.imputation = None