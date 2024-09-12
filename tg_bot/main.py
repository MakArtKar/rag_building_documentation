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
        await message.reply(httpx.get(host, params=query_params).json()['response']['content'])


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
