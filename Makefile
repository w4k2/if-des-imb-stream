all:
	python -W ignore experiment1_GNB.py 1410
	python -W ignore experiment1_HT.py 1410
	python -W ignore experiment1_MLP.py 1410
	python gather.py
	python analyze_1.py
