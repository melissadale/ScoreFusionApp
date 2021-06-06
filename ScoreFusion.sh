#!/bin/bash
mkdir ScoreFusion
cd ScoreFusion

conda create --name sf_environment python=3.85
source activate sf_environment


echo "************************************************************"
echo "****   Cloning ScoreFusionApp from GitHub"
echo "************************************************************"
git clone https://github.com/melissadale/ScoreFusionApp.git .

echo "************************************************************"
echo "****   Installing requirements for ScoreFusionApp"
echo "************************************************************"

pip install -r requirements.txt

echo "************************************************************"
echo "****   All DONE!"
echo "****   Thank you for trying the ScoreFusionApp"
echo "****   You are ready to launch the application with: python GUI.py"
echo "************************************************************"