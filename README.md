# Sarcasm-Checker Bot

## Overview

Sarcasm Checker Bot is a Discord bot designed to detect sarcasm in messages using a fine-tuned TinyBERT model. The bot improves its accuracy over time by learning from user feedback.

## Features

- Detect sarcasm in Discord messages.
- Mark messages as sarcastic with the `!Sarcasm` command.
- Collect user feedback through reactions to improve the model.
- Evaluate model accuracy with the `!Evaluate` command.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Sarcasm-Checker.git
    cd Sarcasm-Checker
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the bot:
    ```sh
    python SarcasmBot.py
    ```

## Usage

- Add the bot to your Discord server.
- Use `!Sarcasm` to mark a message as sarcastic.
- React with 👍 or 👎 to provide feedback on the bot's sarcasm detection.
- Use `!Evaluate` to evaluate the model's accuracy.

## Files

- `SarcasmBot.py`: Main bot implementation.
- `Sarcasm Detection.ipynb`: Notebook for initial model training and fine-tuning.

## Dataset

The model is trained using the [Sarcasm on Reddit](https://www.kaggle.com/datasets/danofer/sarcasm) dataset from Kaggle.

## Model

TinyBERT, a smaller and faster version of BERT, is used for sarcasm detection.

## License

This project is licensed under the MIT License.
