all:
	python -W ignore experiment1_GNB.py 1994
	python gather.py
	python analyze_1.py
