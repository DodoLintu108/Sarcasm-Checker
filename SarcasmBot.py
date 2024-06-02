import discord
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import os
import asyncio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv('discord_bot_token')

model_path = 'tinybert_sarcasm_detector'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def detect_sarcasm(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction == 1

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents)

feedback_data_file = 'feedback_data.csv'

if not os.path.exists(feedback_data_file):
    pd.DataFrame(columns=['message', 'label']).to_csv(feedback_data_file, index=False)

user_messages = {}

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    is_sarcastic = detect_sarcasm(message.content)
    if is_sarcastic:
        response = await message.channel.send(f"{message.author.mention} Stop Being Sarcastic You Stupid Fuck")
        await response.add_reaction('👍')
        await response.add_reaction('👎')
        user_messages[response.id] = message.content

    await bot.process_commands(message)

@bot.event
async def on_reaction_add(reaction, user):
    if user == bot.user:
        return

    message = reaction.message
    if reaction.emoji == '👍' or reaction.emoji == '👎':
        original_message = user_messages.get(message.id)
        if not original_message:
            return

        label = 1 if reaction.emoji == '👍' else 0

        feedback_data = pd.read_csv(feedback_data_file)
        new_feedback = pd.DataFrame([{'message': original_message, 'label': label}])
        feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)
        feedback_data.to_csv(feedback_data_file, index=False)

        await fine_tune_model(feedback_data)

@bot.command(name='sarcasm')
async def mark_as_sarcastic(ctx):
    if ctx.message.reference:
        replied_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        original_message = replied_message.content

        feedback_data = pd.read_csv(feedback_data_file)
        new_feedback = pd.DataFrame([{'message': original_message, 'label': 1}])
        feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)
        feedback_data.to_csv(feedback_data_file, index=False)

        await ctx.send(f"Message '{original_message}' marked as sarcastic and added to dataset.")
        
        await fine_tune_model(feedback_data)
    else:
        await ctx.send("Please reply to a message with this command to mark it as sarcastic.")

@bot.command(name='evaluate')
async def evaluate_model(ctx):
    feedback_data = pd.read_csv(feedback_data_file)
    if len(feedback_data) < 2:
        await ctx.send("Not enough data to evaluate the model.")
        return

    texts = feedback_data['message'].tolist()
    labels = feedback_data['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = SarcasmDataset(train_encodings, train_labels)
    val_dataset = SarcasmDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    eval_results = trainer.evaluate()
    
    preds = trainer.predict(val_dataset).predictions
    preds = torch.argmax(torch.tensor(preds), dim=1).numpy()
    accuracy = accuracy_score(val_labels, preds)
    
    await ctx.send(f"Evaluation accuracy: {accuracy}")

@bot.command(name='commands')
async def show_commands(ctx):
    commands_list = """
    **Sarcasm Checker Bot Commands:**
    - `!sarcasm`: Mark a message as sarcastic.
    - `!evaluate`: Evaluate the model's accuracy.
    - `!commands`: Show this command list.
    """
    await ctx.send(commands_list)

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

async def fine_tune_model(feedback_data):
    texts = feedback_data['message'].tolist()
    labels = feedback_data['label'].tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    dataset = SarcasmDataset(encodings, labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(model_path)

bot.run(TOKEN)
