import pandas as pd 
import os  



csv = pd.read_csv("sample_submission.csv",sep=",")

csv_label = csv["Category"].values
csv_filename = csv["Id"].values

justes = [1 if str(filename.split("_")[0]) == str(label).strip() else 0 for filename,label in zip(csv_filename,csv_label)]
print(justes)
sommes = sum(justes)/len(justes)

#print(dico)   mixnet 0.9548849716163729
#print(csv_dico) try resnext101 0.9536898715267403
# voting model 0.9536898715267403
print(sommes)
