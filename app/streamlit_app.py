import os
import numpy as np
import streamlit as st
import random
import importlib  
from webcam import webcam
from PIL import Image 
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import utils as ut

import footer

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# parameter matplotlib
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

tf.disable_eager_execution()

nr_max_faces=20
nc=6

# load Tensorflow
y, xin, keep_prob_input = ut.set_tf_model_graph(nr_max_faces)
sess = tf.Session()
saver = tf.train.Saver()

# mengembalikan bobot file gambar
saver.restore(sess, 'mlmodels/ekspresi-wajah/model_6layers.ckpt')

# Aplikasi Streamlit
st.title("Aplikasi Pendeteksi Ekspresi Wajah")
st.markdown("Pilih Sumber Gambar pada sidebar,  \n pilih Upload atau webcam(konfirmasi izin penggunaan webcam pada browser)")
st.markdown("Aplikasi dapat mendeteksi lebih dari satu wajah(maksimal 20 wajah)\n"
            "dan menampilkan hasil prediksi ekspresi dari setiap wajah")
st.markdown("Pertama aplikasi akan mendeteksi wajah menggunakan OpenCV")
st.markdown("Kemudian aplikasi mengolah gambar menggunakan Tensorflow dan memberi hasil prediksi ekspresi pada gambar.")

source = st.sidebar.selectbox( "Pilih Sumber Gambar ", ('Upload', 'Webcam') )

if source == 'Webcam':
    captured_image = webcam()
elif source == 'Upload':
    file = st.file_uploader("Upload Gambar disini")
    if file is not None:
        captured_image = Image.open(file)
    else :
        captured_image = None


if captured_image is None:
    st.write("Menunggu Gambar")
else:
    st.write("Mendapatkan Gambar dari {}:".format(source.lower()))

    faces, marked_img = ut.get_faces_from_img(np.array(captured_image))
    st.image(marked_img, use_column_width=True)

    st.subheader('Ditemukan {} Wajah pada gambar diatas:'.format(len(faces)) )

    if len(faces):      
        data_orig = np.zeros([nr_max_faces, 48,48])

        nr_faces = min(len(faces), 20)

        # mengubah data wajah pada gambar menjadi vektor
        for i in range(0, nr_faces):
            data_orig[i,:,:] = ut.contrast_stretch(faces[i,:,:])

        # mempersiapkan gambar
        data = ut.preprocess_faces(data_orig)

        result = sess.run([y], feed_dict={xin: data, keep_prob_input: 1.0}) 
        
        for i in range(0, nr_faces):
            with _lock: 
                plt = ut.plot_face(result[0][i], data[i,:])
                st.pyplot(plt)

footer.footer()
