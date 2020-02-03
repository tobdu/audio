export PIPENV_VENV_IN_PROJECT=true

# -------------------------- docker setup ---------------------------- #

# use local filesystem, but containerize execution
IMAGE=mc706/pipenv-3.6
DIR=$(shell pwd)
DOCKER_RUN=docker run -ti -w /app -v $(DIR):/app $(IMAGE)


# -------------------------- main functions -------------------------- #


# downloading the data and setting up a virtual env
setup: bash_dependencies data/genres create_env


# organizing the data, creating melgrams and label encodings
preprocess: data/index/gtzan_genre.csv data/encoding.pkl data/melgrams.pkl


# train model
train:
	$(DOCKER_RUN) pipenv run python classifier/train.py



# -------------------------- sub functions --------------------------- #


# checking and installing bash dependencies if absent
bash_dependencies:
	./scripts/dependencies.sh assert docker


# downloading data if absent
data/genres:
	$(DOCKER_RUN) curl http://opihi.cs.uvic.ca/sound/genres.tar.gz --output data/gztan.tar.gz
	$(DOCKER_RUN) tar zxvf data/gztan.tar.gz -C data


# recreating a python virtual env containing all project dependencies
compile:
	./scripts/dependencies.sh assert pipenv
	pipenv install

# run the compilation process in a docker container
# but create the .venv folder on your local machine
create_env:
	$(DOCKER_RUN) make compile


# creating data index if absent
data/index/gtzan_genre.csv:
	$(DOCKER_RUN) pipenv run python classifier/preprocess/create_index.py


# create label encodings if absent
data/encoding.pkl:
	$(DOCKER_RUN) pipenv run python classifier/preprocess/label_encoding.py


# computing melgrams if absent
data/melgrams.pkl:
	$(DOCKER_RUN) pipenv run python classifier/preprocess/create_melgrams.py




.PHONY: setup run bash_dependencies create_env preprocess
