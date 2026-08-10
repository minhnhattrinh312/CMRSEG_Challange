# rm -rf output/task2_lge/ output/eval/ output/submission.zip
python src/predict.py
python src/calculate_lv_ef.py
python src/calculate_lge_mass.py
# python src/rename4val.py
# cd output
# zip -r submission.zip task1_cine/ task2_lge/
# cd ..