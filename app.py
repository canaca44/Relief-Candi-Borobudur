#Deploy menggunakan flask
from flask import Flask, render_template, request, redirect
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        #Load model
        model = load_model('Xception_Deployss_epoch20.h5')

        #Simpan Gambar
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        #Labeling Model
        label_code = {'Relief  1':0, 'Relief  2':1, 'Relief  3':2, 'Relief  4':3, 'Relief  5':4, 'Relief  6':5, 'Relief  7':6, 'Relief  8':7, 'Relief  9':8, 'Relief  10':9, 'Relief  11':10, 
        'Relief  12':11, 'Relief  13':12, 'Relief  14':13, 'Relief  15':14, 'Relief  16':15, 'Relief  17':16, 'Relief  18':17, 'Relief  19':18, 'Relief  20':19, 'Relief  21':20, 
        'Relief  22':21, 'Relief  23':22, 'Relief  24':23, 'Relief  25':24, 'Relief  26':25,'Relief  27':26, 'Relief  28':27, 'Relief  29':28, 'Relief  30':29, 'Relief  31':30,
        'Relief  32':31, 'Relief  33':32, 'Relief  34':33, 'Relief  35':34}
        label_decode = ['Relief  1', 'Relief  2', 'Relief  3', 'Relief  4', 'Relief  5', 'Relief  6', 'Relief  7', 'Relief  8', 'Relief  9', 'Relief  10', 'Relief  11', 
        'Relief  12', 'Relief  13', 'Relief  14', 'Relief  15', 'Relief  16', 'Relief  17', 'Relief  18', 'Relief  19', 'Relief  20', 'Relief  21', 
        'Relief  22', 'Relief  23', 'Relief  24', 'Relief  25', 'Relief  26','Relief   27', 'Relief  28', 'Relief  29', 'Relief  30', 'Relief  31',
        'Relief  32', 'Relief  33', 'Relief  34', 'Relief  35']

        #Preprocessing
        img = load_img(image_path, color_mode = "rgb", target_size=(150, 150)) 
        img = np.array(img, dtype = 'float32')
        img = preprocess_input(img)
        img = img.reshape(1,150,150,3)

        #Prediksi Gambar 
        ypred = model.predict(img)

        #Deklarasi nama relief, akurasi dan gambarnya
        name = label_decode[np.argmax(ypred)]
        x = np.argmax(ypred)
        Gagal="https://i.postimg.cc/3JDnPrg7/candi-silang.jpg"
        if(name=="Relief  1"):
            source = "https://i.postimg.cc/8PW2ggW8/Relief-1.jpg"
        elif(name=="Relief  2"):
            source = "https://i.postimg.cc/2jP5xmmg/Relief-2.jpg"
        elif(name=="Relief  3"):
            source = "https://i.postimg.cc/wvY1jpvC/Relief-3.jpg"
        elif(name=="Relief  4"):
            source = "https://i.postimg.cc/8P6xhgPK/Relief-4.jpg"
        elif(name=="Relief  5"):
            source = "https://i.postimg.cc/LhKCyh6y/Relief-5.jpg"
        elif(name=="Relief  6"):
            source = "https://i.postimg.cc/pLxxrLYY/Relief-6.jpg"
        elif(name=="Relief  7"):
            source = "https://i.postimg.cc/26qNMbrm/Relief-7.jpg"
        elif(name=="Relief  8"):
            source = "https://i.postimg.cc/43qSK8Rk/Relief-8.jpg"
        elif(name=="Relief  9"):
            source = "https://i.postimg.cc/sDrb90Q3/Relief-9.jpg"
        elif(name=="Relief  10"):
            source = "https://i.postimg.cc/9XGnMtxK/Relief-10.jpg"
        elif(name=="Relief  11"):
            source = "https://i.postimg.cc/G2njw612/Relief-11.jpg"
        elif(name=="Relief  12"):
            source = "https://i.postimg.cc/QdRQ1vCb/Relief-12.jpg"
        elif(name=="Relief  13"):
            source = "https://i.postimg.cc/d31kz9D3/Relief-13.jpg"
        elif(name=="Relief  14"):
            source = "https://i.postimg.cc/5yY9BK6B/Relief-14.jpg"
        elif(name=="Relief  15"):
            source = "https://i.postimg.cc/W4N2W1db/Relief-15.jpg"
        elif(name=="Relief  16"):
            source = "https://i.postimg.cc/WpBq90TN/Relief-16.jpg"
        elif(name=="Relief  17"):
            source = "https://i.postimg.cc/hvTvF3cS/Relief-17.jpg"
        elif(name=="Relief  18"):
            source = "https://i.postimg.cc/zfvfDf82/Relief-18.jpg"
        elif(name=="Relief  19"):
            source = "https://i.postimg.cc/vmfNVrY4/Relief-19.jpg"
        elif(name=="Relief  20"):
            source = "https://i.postimg.cc/bNrVTcYc/Relief-20.jpg"
        elif(name=="Relief  21"):
            source = "https://i.postimg.cc/Bb6pMH0c/Relief-21.jpg"
        elif(name=="Relief  22"):
            source = "https://i.postimg.cc/W3pw1ghL/Relief-22.jpg"
        elif(name=="Relief  23"):
            source = "https://i.postimg.cc/5yYvX3JG/Relief-23.jpg"
        elif(name=="Relief  24"):
            source = "https://i.postimg.cc/TPtWQSYF/Relief-24.jpg"
        elif(name=="Relief  25"):
            source = "https://i.postimg.cc/15FYgx99/Relief-25.jpg"
        elif(name=="Relief  26"):
            source = "https://i.postimg.cc/tJxDVPGr/Relief-26.jpg"
        elif(name=="Relief  27"):
            source = "https://i.postimg.cc/W1jShhWn/Relief-27.jpg"
        elif(name=="Relief  28"):
            source = "https://i.postimg.cc/FHDbzGJr/Relief-28.jpg"
        elif(name=="Relief  29"):
            source = "https://i.postimg.cc/jdWYZkt9/Relief-29.jpg"
        elif(name=="Relief  30"):
            source = "https://i.postimg.cc/QCHwYHG5/Relief-30.jpg"
        elif(name=="Relief  31"):
            source = "https://i.postimg.cc/GmkN5WNM/Relief-31.jpg"
        elif(name=="Relief  32"):
            source = "https://i.postimg.cc/BnB3KPqQ/Relief-32.jpg"
        elif(name=="Relief  33"):
            source = "https://i.postimg.cc/k4kzZ7np/Relief-33.jpg"
        elif(name=="Relief  34"):
            source = "https://i.postimg.cc/DfLM0kf7/Relief-34.jpg"
        elif(name=="Relief  35"):
            source = "https://i.postimg.cc/xdB6jgWk/Relief-35.jpg"

        #Hasil Prediksi
        if(ypred[0][x]>0.8):
            classification = "terdeteksi sebagai {0} dengan akurasi kemiripan sebesar {1} dengan Relief di bawah ini".format(name,ypred[0][x])
            image_paths = source
        else:
            classification = "terdeteksi sebagai bukan Relief Candi Borobudur dengan tingkat kemiripan hanya {0}".format(ypred[0][x])
            image_paths = Gagal
        
        #Pergi ke halaman prediksi
        return render_template('predict.html', prediction=classification, images=image_paths)


if __name__ == "__main__":
    app.run(debug=True)