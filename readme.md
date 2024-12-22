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

## Results and resume

...


