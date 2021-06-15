#!/bin/bash
mkdir ScoreFusion
cd ScoreFusion
conda create --name scorefusion_environment python=3.8.5
conda activate scorefusion_environment
echo "************************************************************"
echo "****   Cloning ScoreFusionApp from GitHub"
echo "************************************************************"
git clone https://github.com/melissadale/ScoreFusionApp.git
echo "************************************************************"
echo "****   Installing requirements for ScoreFusionApp"
echo "************************************************************"
pip install  -r $PWD/ScoreFusionApp/Utilities/requirements.txt
sudo apt-get install libmtdev1
echo "************************************************************"
echo "****   All DONE!"
echo "****   Thank you for trying the ScoreFusionApp"
echo "****   You are ready to launch the application with: python GUI.py"
echo "************************************************************"