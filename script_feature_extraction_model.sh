#!/bin/bash

#python feature_extraction_model.py budgetcalculator 1 1
#python feature_extraction_model.py homeaffordability 1 1
#python feature_extraction_model.py assetallocationcalculator 1 1
#python feature_extraction_model.py careercalculator 1 1

for j in 1 3 5 7 9 11 13 
do
for i in 1 2 
do
python feature_extraction_model.py budgetcalculator $i $i 0.02 $j 1 -b
python feature_extraction_model.py homeaffordability $i $i 0.02 $j 1  -b
python feature_extraction_model.py assetallocationcalculator $i $i 0.02 $j 1  -b
python feature_extraction_model.py careercalculator $i $i 0.02 $j 1  -b

python feature_extraction_model.py budgetcalculator $i $i 0.05 $j 1  -b
python feature_extraction_model.py homeaffordability $i $i 0.05 $j 1  -b
python feature_extraction_model.py assetallocationcalculator $i $i 0.05 $j 1  -b
python feature_extraction_model.py careercalculator $i $i 0.05 $j 1  -b

python feature_extraction_model.py budgetcalculator $i $i 0.1 $j 1  -b
python feature_extraction_model.py homeaffordability $i $i 0.1 $j 1  -b
python feature_extraction_model.py assetallocationcalculator $i $i 0.1 $j 1  -b
python feature_extraction_model.py careercalculator $i $i 0.1 $j 1  -b

python feature_extraction_model.py budgetcalculator $i $i 0.15 $j 1  -b
python feature_extraction_model.py homeaffordability $i $i 0.15 $j 1  -b
python feature_extraction_model.py assetallocationcalculator $i $i 0.15 $j 1  -b
python feature_extraction_model.py careercalculator $i $i 0.15 $j 1  -b

done
done
