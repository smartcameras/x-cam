#!/bin/bash

wget https://github.com/magicleap/SuperGluePretrainedNetwork/archive/refs/heads/master.zip -O SuperGlue.zip

unzip SuperGlue.zip SuperGluePretrainedNetwork-master/assets/* SuperGluePretrainedNetwork-master/models/* SuperGluePretrainedNetwork-master/README.md -d libs/

mv libs/SuperGluePretrainedNetwork-master libs/SuperGlue

rm SuperGlue.zip
