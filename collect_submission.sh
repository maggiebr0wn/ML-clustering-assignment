rm -f hw2_code.zip
jupyter-nbconvert --to pdf "hw2_code/hw2.ipynb"
zip -r hw2_code.zip  "hw2_code/hw2.pdf" "hw2_code/gmm.py" "hw2_code/kmeans.py" "hw2_code/semisupervised.py"
