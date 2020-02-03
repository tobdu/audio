FROM mc706/pipenv-3.6

RUN apt-get update && apt-get install libsndfile-dev -y