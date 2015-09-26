#!/bin/bash

#python feature_extraction_model.py budgetcalculator 1 1
#python feature_extraction_model.py homeaffordability 1 1
#python feature_extraction_model.py assetallocationcalculator 1 1
#python feature_extraction_model.py careercalculator 1 1

for j in `seq 101 500`
do
#python feature_extraction_model.py budgetcalculator 1 2 0.02 5 1 $j -b
#python feature_extraction_model.py homeaffordability 1 2 0.02 5 1 $j  -b
#python feature_extraction_model.py assetallocationcalculator 1 2 0.02 5 1 $j  -b
python feature_extraction_model.py careercalculator 1 2 0.02 0 1 $j  -b


#python feature_extraction_model.py budgetcalculator 1 2 0.05 1 1 $j -b
#python feature_extraction_model.py budgetcalculator 1 2 0.05 10 1 $j -b
#python feature_extraction_model.py homeaffordability 1 2 0.05 1 1 $j  -b
#python feature_extraction_model.py homeaffordability 1 2 0.05 10 1 $j  -b
#python feature_extraction_model.py assetallocationcalculator 1 2 0.05 1 1 $j  -b
#python feature_extraction_model.py assetallocationcalculator 1 2 0.05 10 1 $j  -b
#python feature_extraction_model.py careercalculator 1 2 0.05 1 1 $j  -b
#python feature_extraction_model.py careercalculator 1 2 0.05 10 1 $j  -b
done
