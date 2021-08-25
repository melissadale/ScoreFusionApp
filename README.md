# Status: 1.0 release #
The first version of this application is released. For a full description of this app, how to use it and how to adapt it, consult the user's manual (https://github.com/melissadale/ScoreFusionApp/blob/master/Score_Fusion_App_Documentation.pdf) or one of the wikis listed in the Documentation section below. 

# DOWNLOAD #
There are various ways to acquire the Score Fusion App. Exe executables are given in the Release directory, if the Exe does not execute, we recommend dowloading from source. A sh script is provided in the Utilities directory that describes the necessary steps. 
* Executable: https://github.com/melissadale/ScoreFusionApp/tree/master/Releases. 
* From Source: https://github.com/melissadale/ScoreFusionApp/blob/master/Utilities/ScoreFusion.sh

# APPLICATION OVERVIEW #

## Data and Data Grooming ##

The first tab allows the user to upload the score files, specify the training testing split, normalization techniques, and how to handle missing data:

<img src="https://github.com/melissadale/ScoreFusionApp/blob/master/DocumentationSupport/data.gif" width="600">

## Density Estimates ##

The second tab provides information about the scores in the dataset, including the number of genuine/imposter scores, detected modalities, and the samples sizes in the train, test, and entire score dataset. Additionally, it provides a PDF and histograms of the genuine and imposter scores within each modality:

<img src="https://github.com/melissadale/ScoreFusionApp/blob/master/DocumentationSupport/densities.gif" width="600">

## Fusion  ##

The third tab allows the user to select which fusion rules to apply, specify if verification and/or identification tasks should be anayized, and give a name to the experiment:

<img src="https://github.com/melissadale/ScoreFusionApp/blob/master/DocumentationSupport/fuse.gif" width="600">


## Results ##

The fourth tab displays the results of the fusion experiments to the user:

<img src="https://github.com/melissadale/ScoreFusionApp/blob/master/DocumentationSupport/results.gif" width="600">

# DOCUMENTATION #
1. [Input File Formats](https://github.com/melissadale/ScoreFusionApp/wiki/Input-File-Formats)
This describes the formats and files allowed by the application. 
2. [Errors, debugging, and feature requests](https://github.com/melissadale/ScoreFusionApp/issues/new)   


