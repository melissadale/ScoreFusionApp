This application performs analysis on provided scores. Because analysis 
and this application's performance rely upon correctly interpreting the
data provided, it is very important to understand the file formats the 
application can understand.

**Things that are true, regardless of data format:**

* **Subject IDs**: The application does its best to identify what might 
be a subject id. IF IDs are strictly numeric, that makes detection more 
challenging. *If* Subject IDs are included as column or row headers, 
ensure they are not strictly numeric.

## Matrix Form ##

This score format contains subjects along the rows and columns such that 
genuine scores are along the diagonal, and the impostor scores are off
diagonal. 

**Good things to know:**

* labels are not necessary, scores along the diagonal are *genuine*, off 
diagonal scores are *impostor*
* Each modality should be its own file. That file name will be what 
the modality is referred to as in the application. 

A pictorial example of this format for the NIST Biometric Score Dataset 
(Face x Finger) at [NIST BSSR1](https://www.nist.gov/itl/iad/image-group/nist-biometric-scores-set-bssr1),
 with subject IDs as row and column headers.
 
![BSSR1 Score Format](https://github.com/melissadale/ScoreFusionApp/blob/master/graphics/Wikis/matrixWheaders.png)



**Assumptions:**

 * The order of subjects is consistent along the column and rows
 * There are the same number of rows and columns  
 * IF there are row or column headers identifying the subject id, that 
 id is not in the same format as the scores (i.e. subject ids should not
 be floats or pure integers)
 
## Column Form ##

There are more formats available when using the *Column Format*. Because 
of the many options, there are more requirements to allow the application
to properly assess the data. 

**Important things to know:**

This format is based along the concept that there are modalities along 
the columns, with the last column containing the label(0 impostor, 1 
genuine) **OR** the first 2 columns must be subject ids - in which case,
 genuine impostor labels may be determined by checking if the first two 
 column ids are equal (genuine) or not (imposter). 
 
* Each column must have a header. This is how the application knows 
    modality titles.
* The label for this last column should be titled label 
or class (capitalization does not matter)
*
