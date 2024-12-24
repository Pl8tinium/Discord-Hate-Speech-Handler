# Discord hate speech handler

This repository contains a POC hate speech detection project that was done for the univerity module "Deep Learning" (FH-SWF). The product is a discord bot, that records the audio of the individual users in a discord channel. It then uses foundational models like `wav2vec` to transcribe the audio file into text. This text will then be analyzed for the sentiment of hate speech. In case the user is flagged as a hate speech propagist, measures can be taken. For the first action I consider temporarily muting to be the most user friendly option.

_The goal is to have a tool that promotes more friendly discord sessions_

## Bot usage

### Dependencies

- install [libopus](https://github.com/xiph/opus) for audio preprocessing
- install python packages `pip install -r requirements-bot.txt`

### Generate API token

Create an .env file and supply the an API token for the discord bot (`DISCORD_BOT_TOKEN`). You can checkout how to create a token [here](https://discordpy.readthedocs.io/en/stable/discord.html), but for simplification here are additional hints  and information.

- `Bot` section in the individual application under the [application settings](https://discord.com/developers/applications/)
    - `Privileged Gateway Intents` -> enable all
    - `Public Bot` -> enable
    - `Token` -> Generate an API token
- Generate invite link
    - `Oauth2` section -> `OAuth2 URL Generator` -> `bot` -> `administrator` -> `guild install`

### Usage

Invite the bot via the generated OAuth2 url and run the script `mutant_bot.py`. Join a channel with your normal discord client and make use of the following commands.

- !join -> bot joins the channel
- !record -> bot starts hate speech detection
- !stop -> bot stops hate speech detection

## Research

The sentiment analysis is done via an already fine tuned foundational model from [deepset](https://huggingface.co/deepset/bert-base-german-cased-hatespeech-GermEval18Coarse). Though this model has been trained with a specific dataset, that is already a few years old at this point (GermEval18 -> 2018). I did find larger datasets for hate speech classification, which made me wonder how good the pretrained model actually performs and if I could train something that maybe even performs better.

### Dataset

The additional dataset I found was a superset that aggregated multiple hate-speech datasets into one. Thereby making the largest bundled size of german hate speech available. The dataset is from [manueltonneau](https://huggingface.co/datasets/manueltonneau/german-hate-speech-superset) and contains ~50k samples. On the other hand the GermEval18 dataset only contains ~8.5k samples.

### Foundational model

As a foundational model I have choosen _dbmdz's_ [bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased). I have choosen a BERT model, because I thought that it may be interesting to use the same underlying network structure as the pretrained one, to figure out if more data actually makes the difference. Ofcourse, `bert-base-german-uncased` is also already fine tuned for the german language and it may have been finetuned differently/ better than _deepsets_ model in that realm, but I thought its still a fair comparison.

## The process

To get to the final results of the model, it was not so straightforward. I faced some issues that had to be analyzed and fixed, until i was satisfied with the results of the predictions.

The main problem i faced throughout the various training runs that my loss was essentially never falling below _0.5_. I was able to fix this at the end. The various techniques implemented in the code are now listed in the order i implemented them.

### Adding dropouts for generalization

Initially my training runs looked good and in relation of the low amount of epochs and smaller initial training set. I thought not being under a loss of _0.5_ would simply be solved by more epochs and just scaling up. At this stage i noticed another problem though. In my early test runs the train and test set always diverged _by a lot_ the longer my training runs have gotten.

Thats why I decided to look into what i could do to generalize more. I came to the conclusion that dropouts may be a good choice for that. They did help a little bit and contributed to the larger puzzle, but initially no _great_ progress was noticable.

### Adding more layers for better learning capabilities

I played around with a lot of parameters, switched out the datasets. Did longer runs and so on. Then I noticed the big issue, which was explained in the beginning of this chapter. It seamed like I was not able to get below a loss of _0.5_. I thought that only having one linear layer concatinated at the output of the frozen BERT model may not have been enough, which made me wonder if adding more simple feed forward layers would maybe improve the learning the model would be capable of.

At this stage I added linear layers with batch normalization and a dynamic learning rate process to ensure that when the model goes down the right path it stays in this path. I even executed the model for a bit more than 24 hours, but did not notice a great improvement either. I was not able to get below 0.5.

![mid training run](./doc/img/training_run_img_1.png)

### Priorizite Precision

I noticed a third issue, or better said I maybe noticed the reason for the learning wall I seam to have hit. It looked like that my model had a great or okaish accuracy in determining which sample to flag as hate speech. On the countrary my precision, the metric that tells me how many false examples have mistakenly being flagged as positive, was _extreeemely_ low (0.06 in one of the early training runs to be exact). I decided to implement techniques that would help me to not flag things as positive if they are actually negative. Thereby I came up and integrated the following techniques.

- added weighted classes (getting the class 0, no hatespeech, wrong, would penalize the model/ the loss harder)
- balance out training set (equal amount of true hatespeech cases and no hatespeech cases)
- cleaned training data (remove unnecessary characters and normalize the text)

I did notice a great improvement with the appliance of the techniques as can be seen in the graph below.

![late training run](./doc/img/training_run_img_2.png)

Also my metrics have substantially improved as can be seen here. I now have hit a precision of 0.72! I was very happy about the results.

![late training run metrics](./doc/img/training_run_img_2_metrics.png)

### GPU rich

While doing the experiment, where I tried to bump up the precision, I also did another experiment. I tried to use the larger dataset (hatespeech superset, not GermEval18) and just increase the epochs by a substantial amount. On my local machine it was a bit slow, but luckily I was able to use the compute cluster of my school, which sped up the process and allowed me to run the model for a huge number of epochs for a little bit longer than a day. The amount of data in comparison to the GermEval18 set I was using before was roughly _x6_.

I got great results out of that. Finally, my loss got down to _~0.25_.

![large training run](./doc/img/training_run_img_3.png)

### The final run 

The pipeline that was used here was very plain in comparison how I tweaked it with the precision techniques. That's why I thought that even though the results are already satisfying enough, I wanted so see what both approaches would yield when used together.

![]

XXXXXXXXXXXX

## Results

The model performs XXXXXXXXXX

## Resume

The project was quite fun and I learned a lot. I think I really have a new tool in my toolbox as a programmer. I feel equiped to research and implement various neural net architectures and make use of them. This will immensely help me when I face problems that are very difficult to solve over the conventional deterministic programming way.