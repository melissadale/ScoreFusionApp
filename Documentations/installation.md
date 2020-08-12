# Detailed walk through of installation # 

This application has been developed on Windows 10 machines, the 
following outlines a detailed process of getting this application set up
. These instructions have only been vetted on a few windows machines, 
and there is a chance that you will need to make alterations to ensure
proper functioning on your own machine. 

**Tools**: 

* **Python**: https://www.python.org/downloads/windows/ (NOTE: Please check 
"add to PATH" checkbox in the bottom of the installation screen
* **Git** (Optional):
https://git-scm.com/downloads
* **Anaconda** (Optional): 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html

**From PowerShell:**

1. Navigate to the directory you would like the code to live
2. Run `git clone https://github.com/melissadale/ScoreFusionApp.git .`
    * If git is not installed, you can download the hard copy of the 
    code directly from github and navigating to the directory in 
    PowerShell (or command prompt)
3. (Optional, but recommended) Create virtual environment. If using 
Conda, then run `conda create --name scorefuse_env` and `activate 
scorefuse_env`
4. Run `pip install -r requirements.txt`
5. Run `python GUI.py`