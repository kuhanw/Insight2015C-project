#!/bin/bash

#python feature_extraction_model.py budgetcalculator 1 1
#python feature_extraction_model.py homeaffordability 1 1
#python feature_extraction_model.py assetallocationcalculator 1 1
#python feature_extraction_model.py careercalculator 1 1

for j in `seq 707 1000`
do
python feature_extraction_model.py -w budgetcalculator -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j 
python feature_extraction_model.py -w homeaffordability -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j 
python feature_extraction_model.py -w assetallocationcalculator -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j 
python feature_extraction_model.py -w careercalculator -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j 


#python feature_extraction_model.py budgetcalculator 1 2 0.05 1 1 $j -b
#python feature_extraction_model.py budgetcalculator 1 2 0.05 10 1 $j -b
#python feature_extraction_model.py homeaffordability 1 2 0.05 1 1 $j  -b
#python feature_extraction_model.py homeaffordability 1 2 0.05 10 1 $j  -b
#python feature_extraction_model.py assetallocationcalculator 1 2 0.05 1 1 $j  -b
#python feature_extraction_model.py assetallocationcalculator 1 2 0.05 10 1 $j  -b
#python feature_extraction_model.py careercalculator 1 2 0.05 1 1 $j  -b
#python feature_extraction_model.py careercalculator 1 2 0.05 10 1 $j  -b
done
