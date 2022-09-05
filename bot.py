from config import token
import telebot
from Model import model, evaluate, char_to_idx, idx_to_char

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Начинаю финтить.')

@bot.message_handler(content_types=['text'])
def message_reply(message):
    model.eval()
    msg = evaluate(
        model,
        char_to_idx,
        idx_to_char,
        temp=0.3,
        prediction_len=128,
        start_text=message.text
    )
    bot.reply_to(message, msg)

if __name__ == "__main__":
    bot.polling(none_stop=True, interval=0)
