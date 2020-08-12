# Status: *INPROGRESS* #
This application is in active development. There are known bugs and more 
functionality in the works. 

# USAGE #

**Setup to run on your machine**:

This application will be packaged as an executable in the future. For 
the current time, please use the following instructions, which assumes 
you have pip and python installed:

1. In a terminal, navigate to the directory containing the application's
code
2. Run `pip install -r requirements.txt`
3. Launch the GUI with `python GUI.py`


**Assumptions**:
- all scores are in one directly (no sub directories)


# DEVELOPMENT NOTES # 
This section is dedicated to people who may working on the development 
and progression of the application. It describes how to set up the coding
environment properly. A **user** of the application should follow the above 
section. 

At times, setting up Kivy for the first time can be challenging. For
instance on Windows, the order that packages are installed matters. 

Installing Kivy to python environment the first time, it is a good idea
to follow:(StackOverflow Post)[https://stackoverflow.com/questions/49482753/sdl2-importerror-dll-load-failed-the-specified-module-could-not-be-found-and]

**In summary**, run the following
```
python -m pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew --extra-index-url https://kivy.org/downloads/packages/simple/
```

**If updating requirements.txt**, generate with pipreqs, move kivy to the bottom and add the following before it
```
wheel==0.34.2
setuptools==46.1.3.post20200325
docutils==0.16
pygments==2.6.1
pypiwin32==223
kivy.deps.sdl2==0.2.0
kivy.deps.glew==0.2.0
--extra-index-url https://kivy.org/downloads/packages/simple/
```

Note on version numbers, these were taken from a currently working system. If needed to determine which version is installed, use `pip show package` to see information. 


# Documentation #
1. [Input File Formats](https://github.com/melissadale/ScoreFusionApp/blob/master/Wikis/FileFormats.md)
This describes the formats and files allowed by the application. 
2. [Errors, debugging, and feature requests](https://github.com/melissadale/ScoreFusionApp/issues/new)   
