# **EC Industry Classifier – Quick-Start Guide**

###### 

###### Author Nicolas Thollot · nicolas.thollot@endeavor.org

###### Usage Internal only. Redistribution requires prior written consent.



-------------------------------------------------------------------------------------



### **Overview**



###### This document outlines the minimal workflow for training and using the EC Industry Classifier.

###### If you already have a pretrained model (\*.joblib), skip directly to Step 3.



-------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------



##### Step 1 – Assemble Your Training Dataset



Create an Excel workbook (.xlsx) with one sheet that contains the following columns in exactly this order:



company , label , label2 , description



We recommend at least eight (8) companies per sub-industry (label2) to ensure reliable model performance.



NOTE: Descriptions MUST be in English. You may use the file *translate.exe* if you want to automatically translate all descriptions in your xlsx file. 





##### Step 2 – Train the Model



Run the file *train\_model.exe* from your folder.



The program will prompt you for:



* Path to the Excel file
* Sheet name or index
* Output location for the resulting *hierarchical\_classifier.joblib*



Training progress and validation statistics are displayed in-line.





##### Step 3 – Generate Predictions (Optional for Pretrained Models)



Run the file *predict\_model.exe*. 



You will be asked to provide:



* The path to the .joblib model
* An Excel file containing the companies to classify
* Column names for the company identifier and description
* (Optional) API keys and model selections for OpenAI or Gemini LLM “second-opinion” predictions



If you want to select multiple models, simply comma separate the numbers attributed to the models you'd like to use (e.g: 2,10,32)



Your final spreadsheet will be outputted and contain the columns and confidence scores for each model's prediction. (broad = broad industry, spec = specific industry within the broad industry).





