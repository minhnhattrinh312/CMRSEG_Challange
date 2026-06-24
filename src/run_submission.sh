rm -rf output/task2* output/*.zip
python src/predict.py
# python src/calculate_lv_ef.py
python src/calculate_lge_mass.py
python src/rename4val.py