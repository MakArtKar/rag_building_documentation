import os
import subprocess

import asyncio
import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from dotenv import load_dotenv

load_dotenv()
TG_API_TOKEN = os.getenv('TG_API_TOKEN')
ANSWERING_HOST = os.getenv('ANSWERING_HOST')
DOWNLOAD_DIR = os.getenv('DOWNLOAD_DIR')
RETRIEVER_URL = os.getenv("RETRIEVER_URL")
HTTP_TIMEOUT = 1200

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

bot = Bot(token=TG_API_TOKEN)
dp = Dispatcher()


@dp.message(Command(commands=['start', 'help']))
async def send_welcome(message: types.Message):
    await message.reply("Hi!")


@dp.message(Command('ping'))
async def send_pong(message: types.Message):
    try:
        pong = "Bot is up!"
    except subprocess.CalledProcessError:
        pong = "Bot is down!"

    await message.reply(pong)


@dp.message(Command('ask'))
async def ask(message: types.Message):
    user_question = message.text[len('/ask'):].strip()

    if user_question == '':
        await message.reply('No question provided!')
    else:
        host = os.path.join(ANSWERING_HOST, 'ask')
        query_params = {
            'query': user_question,
            'num': 5,
        }
        response = httpx.get(host, params=query_params, timeout=HTTP_TIMEOUT)
        if response.status_code != 200:
            await message.reply("Smth went wrong...")
        else:
            await message.reply(response.json()['response']['content'])


@dp.message(lambda message: message.document)
async def handle_document(message: types.Message):
    document = message.document

    file_id = document.file_id
    file_name = document.file_name

    file = await bot.get_file(file_id)
    file_path = file.file_path

    destination_file = os.path.join(DOWNLOAD_DIR, file_name)
    await bot.download_file(file_path, destination_file)

    file = {'file': open(destination_file, 'rb')}
    response = httpx.post(os.path.join(RETRIEVER_URL, 'upload'), files=file, timeout=HTTP_TIMEOUT)

    if response.status_code == 200:
        await message.answer(f"Document {file_name} has been downloaded successfully.")
    else:
        await message.answer(f"Something went wrong...")


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
