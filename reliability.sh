#!/bin/bash

#python feature_extraction_model.py budgetcalculator 1 1
#python feature_extraction_model.py homeaffordability 1 1
#python feature_extraction_model.py assetallocationcalculator 1 1
#python feature_extraction_model.py careercalculator 1 1

for j in 1
do
for i in 2 
do
python feature_extraction_model.py budgetcalculator 1 $i 0.01 $j 1 -b
python feature_extraction_model.py homeaffordability 1 $i 0.01 $j 1  -b
python feature_extraction_model.py assetallocationcalculator 1 $i 0.01 $j 1  -b
python feature_extraction_model.py careercalculator 1 $i 0.01 $j 1  -b

python feature_extraction_model.py budgetcalculator 1 $i 0.05 $j 1  -b
python feature_extraction_model.py homeaffordability 1 $i 0.05 $j 1  -b
python feature_extraction_model.py assetallocationcalculator 1 $i 0.05 $j 1  -b
python feature_extraction_model.py careercalculator 1 $i 0.05 $j 1  -b

python feature_extraction_model.py budgetcalculator 1 $i 0.1 $j 1  -b
python feature_extraction_model.py homeaffordability 1 $i 0.1 $j 1  -b
python feature_extraction_model.py assetallocationcalculator 1 $i 0.1 $j 1  -b
python feature_extraction_model.py careercalculator 1 $i 0.1 $j 1  -b

python feature_extraction_model.py budgetcalculator 1 $i 0.15 $j 1  -b
python feature_extraction_model.py homeaffordability 1 $i 0.15 $j 1  -b
python feature_extraction_model.py assetallocationcalculator 1 $i 0.15 $j 1  -b
python feature_extraction_model.py careercalculator 1 $i 0.15 $j 1  -b

done
done
