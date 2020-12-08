#!/usr/bin/env sh

REFERENCEFILE=captions_val2014.json
LOCREF=cocoeval/annotations
METEORFILE=paraphrase-en.gz
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."
wget https://raw.githubusercontent.com/alhparsa/show-and-tell/master/cocoeval/annotations/captions_val2014.json
wget https://github.com/alhparsa/show-and-tell/raw/master/cocoeval/pycocoevalcap/meteor/data/paraphrase-en.gz
mkdir cocoeval/pycocoevalcap/meteor/data
mv $METEORFILE cocoeval/pycocoevalcap/meteor/data/$METEORFILE
mv $REFERENCEFILE $LOCREF/$REFERENCEFILE
echo "Done."

