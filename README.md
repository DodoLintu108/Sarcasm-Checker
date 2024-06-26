# Sarcasm-Checker Bot

## Overview

Sarcasm Checker Bot is a Discord bot designed to detect sarcasm in messages using a fine-tuned TinyBERT model. The bot improves its accuracy over time by learning from user feedback.

## Features

- Detect sarcasm in Discord messages.
- Mark messages as sarcastic with the `!sarcasm` command.
- Collect user feedback through reactions to improve the model.
- Evaluate model accuracy with the `!evaluate` command.
- List available commands with the `!commands` command.

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

3. Create a `.env` file with your bot token:
    ```sh
    echo "DISCORD_BOT_TOKEN=your_discord_bot_token_here" > .env
    ```

4. Run the bot:
    ```sh
    python SarcasmBot.py
    ```

## Usage

- Add the bot to your Discord server.
- Use `!sarcasm` to mark a message as sarcastic.
- React with 👍 or 👎 to provide feedback on the bot's sarcasm detection.
- Use `!evaluate` to evaluate the model's accuracy.
- Use `!commands` to list all available commands.

## Files

- `SarcasmBot.py`: Main bot implementation.
- `Sarcasm Detection.ipynb`: Notebook for initial model training and fine-tuning.

## Dataset

The model is trained using the [Sarcasm on Reddit](https://www.kaggle.com/datasets/danofer/sarcasm) dataset from Kaggle.

## Model

TinyBERT, a smaller and faster version of BERT, is used for sarcasm detection.

## License

This project is licensed under the MIT License.
