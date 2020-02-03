export PIPENV_VENV_IN_PROJECT=true



# -------------------------- docker setup ---------------------------- #

# use local filesystem, but containerize execution
IMAGE=audio
DIR=$(shell pwd)
DOCKER_RUN=docker run -ti -w /app -v $(DIR):/app $(IMAGE)





# -------------------------- main functions -------------------------- #

# run everything
all: setup preprocess train


# downloading the data and setting up a virtual env
setup: bash_dependencies docker_build data/genres .venv


# organizing the data, creating melgrams and label encodings
preprocess: data/index/gtzan_genre.csv data/encoding.pkl data/melgrams.pkl


# train model
train:
	$(DOCKER_RUN) pipenv run python classifier/train.py






# -------------------------- setup ----------------------------------- #



# checking and installing bash dependencies if absent
bash_dependencies:
	./scripts/dependencies.sh assert docker

docker_build:
	docker build -t $(IMAGE) .

# downloading data if absent
data/genres:
	$(DOCKER_RUN) curl http://opihi.cs.uvic.ca/sound/genres.tar.gz --output data/gztan.tar.gz
	$(DOCKER_RUN) tar zxvf data/gztan.tar.gz -C data


# run the compilation process in a docker container
# but create the .venv folder on your local machine
.venv:
	$(DOCKER_RUN) pipenv --three install





# -------------------------- preprocessing --------------------------- #


# creating data index if absent
data/index/gtzan_genre.csv:
	$(DOCKER_RUN) pipenv run python classifier/preprocess/create_index.py


# create label encodings if absent
data/encoding.pkl:
	$(DOCKER_RUN) pipenv run python classifier/preprocess/label_encoding.py


# computing melgrams if absent
data/melgrams.pkl:
	$(DOCKER_RUN) pipenv run python classifier/preprocess/create_melgrams.py









.PHONY: setup train bash_dependencies docker_build preprocess all
