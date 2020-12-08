## Source code for Iron Racoons Final Project.
We highly recommend using the colab link for setting up the project and running the training instance.

https://colab.research.google.com/drive/1kF0dM_OWVidVihNLpxLp5W27FSD_yGyM?usp=sharing

Everything is set up there and you can simply run each cell. If you decide to run everything locally 
follow the steps below.

## Downloading the data
Download the training and validation set using the links below:

http://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip

http://s3.us-east-1.amazonaws.com/images.cocodataset.org/zips/train2014.zip

http://s3.us-east-1.amazonaws.com/images.cocodataset.org/zips/val2014.zip

http://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip

After downloading the files, unzip them all into a folder called `data` within your current directory.

Your directory structure should be like below after unzipping the data:
```
show-and-tell/
              data/
                  annotations/
                  train2014/
                  val2014/
```

## Weights
All of our trained weights are available to be download from the following link:
https://drive.google.com/drive/folders/16GKgOIUhUUr7z7gP8RFThPeWjEuzN0oX?usp=sharing

## Setting up
Once you are done with unzipping, make sure you have `nltk` library installed and have the tokenizer `punkt` downloaded if not, use the following code in python to download the tokenizer:

```
import nltk
nltk.download('punkt')
```
Then run `python fetch_data.py` to dump all tokens in the training annotation dataset.

## Training
For training you can simply run `python train.py` to start training. By default, it will train a model from scratch, with batch_size of 256 using the `Show Tell and Attend` Model. To train a `Show and Tell` model set the parameter `--with_attention` to false when running in the commandline. There are more parameters which you can modify. After each epoch of training, the program generates a `.ckpt` at the current directory, but you can modify the location by setting the `--checkpoint_dir` parameter.

## Caption Generation
For generating captions for validation dataset for evaluation you can run `python caption_genration.py`, the program requires a path to the trained model which is specified by `--checkpoint_file`, the batch_size (the batch_size for show attend and tell must be set to 1) and for show attend and tell you must set the parameter `--with_attention` to true in the command line. 

## Model evaluation
To evaluate the models, we used the coco-evaluator program, which resides in `cocoeval` folder. Before doing any evalution make sure you run the `cocoeval/get_stanford_models.sh` program. Then using **python2** run `evaluateModel.py`. By default it will evaluate the Show Attend and Tell generated captions, but you can change it by setting the `--file_path` to the generated json file.

## Code Description
To start off we modified [muggin's Show and Tell](https://github.com/muggin/show-and-tell) model and made changes based on our needs. This code was used as a base for building off our show attend and tell model. We use the main components of [sgrvinod's Show Attend and Tell](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) to build off our show attend and tell model. We debugged quite few bugs from muggin's code and made changes for what we needed, and integrated sgrvinod's model on top of that. Similarly, we made changes to sgrvinod to reflect what we wanted and also to be able to make direct comparisions with the Show and Tell model. 

## Acknowledgments
- [muggin's Show and Tell](https://github.com/muggin/show-and-tell)
- [sgrvinod's Show Attend and Tell](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- [AaronCCWong's Show Attend and Tell](https://github.com/AaronCCWong/Show-Attend-and-Tell)