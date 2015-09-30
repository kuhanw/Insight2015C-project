#!/bin/bash

#python FeatureRank.py budgetcalculator 1 1
#python FeatureRank.py homeaffordability 1 1
#python FeatureRank.py assetallocationcalculator 1 1
#python FeatureRank.py careercalculator 1 1

for j in `seq 101 1000`
do
python FeatureRank.py -json web_text_v12_data_set_1_2.json -w budgetcalculator -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j -f 0.25 
python FeatureRank.py -json web_text_v12_data_set_1_2.json -w homeaffordability -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j -f 0.25 
python FeatureRank.py -json web_text_v12_data_set_1_2.json -w assetallocationcalculator -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j -f 0.25
python FeatureRank.py -json web_text_v12_data_set_1_2.json -w careercalculator -nL 1 -nH 2 -mDF 0.02 -pLoad 5 -wView 1 -Seed $j -f 0.25 

#python FeatureRank.py -json web_text_v12_data_set_1_2.json -w budgetcalculator -nL 1 -nH 2 -mDF 0.05 -pLoad 5 -wView 1 -Seed $j 
#python FeatureRank.py -json web_text_v12_data_set_1_2.json -w homeaffordability -nL 1 -nH 2 -mDF 0.05 -pLoad 5 -wView 1 -Seed $j 
#python FeatureRank.py -json web_text_v12_data_set_1_2.json -w assetallocationcalculator -nL 1 -nH 2 -mDF 0.05 -pLoad 5 -wView 1 -Seed $j 
#python FeatureRank.py -json web_text_v12_data_set_1_2.json -w careercalculator -nL 1 -nH 2 -mDF 0.05 -pLoad 5 -wView 1 -Seed $j 
#python FeatureRank.py budgetcalculator 1 2 0.05 1 1 $j -b
#python FeatureRank.py budgetcalculator 1 2 0.05 10 1 $j -b
#python FeatureRank.py homeaffordability 1 2 0.05 1 1 $j  -b
#python FeatureRank.py homeaffordability 1 2 0.05 10 1 $j  -b
#python FeatureRank.py assetallocationcalculator 1 2 0.05 1 1 $j  -b
#python FeatureRank.py assetallocationcalculator 1 2 0.05 10 1 $j  -b
#python FeatureRank.py careercalculator 1 2 0.05 1 1 $j  -b
#python FeatureRank.py careercalculator 1 2 0.05 10 1 $j  -b
done
