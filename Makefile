export PIPENV_VENV_IN_PROJECT=true


# -------------------------- main functions -------------------------- #


# downloading the data and setting up a virtual env
setup: bash_dependencies data/genres create_env


# organizing the data, creating melgrams and label encodings
preprocess: data/index/gtzan_genre.csv data/encoding.pkl data/melgrams.pkl


# train model
train:
	pipenv run python classifier/train.py



# -------------------------- sub functions -------------------------- #


# checking and installing bash dependencies if absent
bash_dependencies:
	./scripts/dependencies.sh assert brew tar curl
	./scripts/dependencies.sh install pipenv


# downloading data if absent
data/genres:
	curl http://opihi.cs.uvic.ca/sound/genres.tar.gz --output data/gztan.tar.gz
	tar zxvf data/gztan.tar.gz -C data


# recreating a python virtual env containing all project dependencies
create_env:
	pipenv --three install


# creating data index if absent
data/index/gtzan_genre.csv:
	pipenv run python classifier/preprocess/create_index.py


# create label encodings if absent
data/encoding.pkl:
	pipenv run python classifier/preprocess/label_encoding.py


# computing melgrams if absent
data/melgrams.pkl:
	pipenv run python classifier/preprocess/create_melgrams.py




.PHONY: setup run bash_dependencies create_env preprocess
