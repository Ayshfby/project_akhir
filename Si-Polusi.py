import discord
from discord.ext import commands
import uuid
from keras.preprocessing import image
from PIL import Image
from keras.models import load_model
import numpy as np


# Variabel intents menyimpan hak istimewa bot
intents = discord.Intents.default()
# Mengaktifkan hak istimewa message-reading
intents.message_content = True
# Membuat bot di variabel klien dan mentransfernya hak istimewa
bot = commands.Bot (command_prefix="$", intents=intents)

def get_class(model_path, labels_path, image_path):
    # load model
    model = load_model(model_path)

    # load label
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image

    # Inferensi
    predictions = model.predict(img_array)
    index = np.argmax(predictions)


    predict_idx = np.argmax(predictions, axis=1)
    predict_label = labels[predict_idx[0]]
    predict_score = predictions[0][index]

    return predict_label,predict_score

@bot.event
async def on_ready():
    print(f'Kita telah masuk sebagai {bot.user}')

@bot.command()
async def halo_bot(ctx):
    await ctx.send("Hallo, selamat datang di bot si polusi")
    await ctx.send("Apakah yang ingin ditanyakan?")

@bot.command()
async def tanya(ctx):
    await ctx.send("Tentu saja aku bisa!!")
    await ctx.send("silakan unggah foto kendaraan yang dipunya menggunakan command ($mengunggah_foto)")

@bot.command(name='mengunggah_foto')
async def mengunggah_foto(ctx):
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]

        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            unique_filename = f"{uuid.uuid4()}_{attachment.filename}"
            file_path = f"photos/{unique_filename}"

            await attachment.save(file_path)

            # Proses integrasi AI
            model_path = "converted_keras/keras_model.h5"
            labels_path = "converted_keras/labels.txt"
            result,score = get_class(model_path,labels_path, file_path)

            # Inform the user that the image has been saved
            await ctx.send(f"Image saved as {unique_filename}")

            # Inform user the result from AI
            await ctx.send(f"Your AI Predict: {result}")
            print(result)
            await ctx.send(f"Your score accuration: {round(score*100,2)}%")

            if result == "0 polusi paling tinggi":
                await ctx.send("!!!kami punya beberapa solusi untuk itu!!!")
                await ctx.send("sebelumnya, kendaraan yang kamu gunakan sangat berdampak buruk untuk lingkungan sekitar mu!!")
                await ctx.send("""ini dia beberapa masukan yang dapat kamu ambil:
                 1. kamu harus mulai mencari alternatif lain saat ingin berkendara sperti menggunakan kendaraan umum agar tidak terjadi seperti itu
                 2. gunakan kendaraan tersebut hanya jika benar benar diperlukan dan tidak ada alternatif lain
                 3. dan, jika kamu harus menggunakan kendaraan tersebut gunakan lah kendaraan yang tidak menghasilkan polusi atau yang berpolusi rendah seperti truk listri, mobil listrik, dan sepeda motor listrik""")
            
            if result == "1 polusi tinggi":
                 await ctx.send("!!!kami punya beberapa solusi untuk itu!!!")
                 await ctx.send("sebelumnya, kendaraan yang kamu gunakan cukup berdampak buruk bagi lingkungan mu!!")
                 await ctx.send("""ini dia beberapa masukan yang dapat kamu ambil: 
                 1. Kamu bisa mengganti kendaraan mu dengan menggunakan kendaran yang berpolusi rendah, atau yang tidak menghasilkan polusi sama sekali, seperti mobil yang menggunakan listrik
                 2. alternatif lainnya juga, kamu dapat menggunakan kendaraan umum seperti kereta, maupun bus untuk berpergian ke tempat tertentu""")
            
            if result == "2 polusi sedang":
                await ctx.send("!!!kami punya beberapa solusi untuk itu!!!")
                await ctx.send("sebelumnya, kamu sudah berkontribusi untuk mengurangi polusi di sekitar mu, dengan menggunakan kendaraan berpolusi rendah")
                await ctx.send("""tapi, itu saja belum cukup, ini dia beberapa saran dari si polusi:
                 1. jika kamu ingin pergi dan jarak tujuannya tidak jauh, lebih baik berjalan kaki
                 2. kamu juga bisa menggunakan kendaraan yang tidak menghasilkan polusi, seperti sepeda kayuh, maupun sepeda listrik 
                 3. last but not least kamu juga bisa menggunakan kendaraan umum, hal itu bisa sangat berdampak terhadap polusi yang dihasilkan!!""")

            if result == "3 polusi 0":
                 await ctx.send("!!! Wauu, kamu sudah berperan besar sekali dalam mengurangi polusi di lingkungan sekitar mu !!!")
                 await ctx.send("terus pertahankan hal tersebut ya")

        else:
            # Inform the user that the attachment is not an image
            await ctx.send("The attached file is not an image. Please upload a .png, .jpg, .jpeg, or .gif file.")
    else:
        # Inform the user that no attachment was found
        await ctx.send("No attachment found in the message. Please upload an image.")

bot.run("Token")
            #file_path = f"photos/{unique_filename}"

            #await attachment.save(file_path) 
