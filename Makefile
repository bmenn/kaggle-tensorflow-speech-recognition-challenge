# Requires presence of credentials.txt file containing login/password in the following format:
# UserName=my_username&Password=my_password

COMPETITION=tensorflow-speech-recognition-challenge

all: download_files

session.cookie: credentials.txt
	curl -c session.cookie https\://www.kaggle.com/account/login
	curl -c session.cookie -b session.cookie -L -d @credentials.txt https\://www.kaggle.com/account/login

files.txt: session.cookie
	curl -c session.cookie -b session.cookie -L http\://www.kaggle.com/c/$(COMPETITION)/data | \
	grep -o \"[^\"]*\/download[^\"]*\" | sed -e 's/"//g' -e 's/^/http:\/\/www.kaggle.com/' > files.txt

download_files: files.txt session.cookie
	mkdir -p data/raw
	cd data/raw && xargs -n 1 curl --limit-rate 4M -b ../../session.cookie -L -O < ../../files.txt

.PHONY: clean

clean:
	rm session.cookie files.txt data/raw/*
