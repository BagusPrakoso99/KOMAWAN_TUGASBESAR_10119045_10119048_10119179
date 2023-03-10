import cv2
import matplotlib.pyplot as plt
import PIL 
import numpy as np
import tensorflow.compat.v1 as tf

from scipy.sparse import coo_matrix

nc=6

str_emotions = ['Marah','Takut','Senang','Sedih','Terkejut','Normal']

# membuat array wajah yang terdeteksi 
def get_faces_from_img(image):
    
    cascPath = 'opencv-artifacts/haarcascade_frontalface_default.xml'
    
    faceCascade = cv2.CascadeClassifier(cascPath)

    # konversi wajah ke greyscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # deteksi wajah pada gambar
    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # persiapan array untuk setiap wajah yang terdeteksi 
    faces_imgs = np.zeros((len(faces),48,48))

    # memisahkan tiap array untuk wajah yang terdeteksi
    num_fac = 0

    for (x, y, w, h) in faces:

        face_single = image[y:y+h,x:x+w]
        # resize gambar menjadi 48x48
        face_resized = cv2.resize(face_single, (48,48))
        faces_imgs[num_fac] = face_resized[:,:,0]
        num_fac = num_fac+1

    # memberi tanda pada wajah yang terdeteksi dalam gambar
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return faces_imgs, image

def convert_to_one_hot(a,max_val=None):
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())

def convert_from_one_hot(a):
     return np.argmax(a);
    
def get_min_max( img ):
    return np.min(np.min(img)),np.max(np.max(img))

def remap(img, min, max):
    if ( max-min ):
        return (img-min) * 1.0/(max-min)
    else:
        return img
    
# fungsi kontras pada gambar 
def contrast_stretch(img ):
    min, max = get_min_max( img );
    return remap(img, min, max)

def set_tf_model_graph(nr_max_faces):
    d = 2304 

    def weight_variable2(shape, nc10):
            initial2 = tf.random_normal(shape, stddev=tf.sqrt(2./tf.to_float(ncl0)) )
            return tf.Variable(initial2)
        
    def conv2dstride2(x,W):
            return tf.nn.conv2d(x,W,strides=[1, 2, 2, 1], padding='SAME')

    def conv2d(x,W):
            return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')
        
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=1/np.sqrt(d/2) )
            return tf.Variable(initial)
        
    def bias_variable(shape):
        initial = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)
    
    tf.reset_default_graph()

    # hasil grafik
    batch_size = nr_max_faces    
    nc = 6                  

    xin = tf.placeholder(tf.float32,[batch_size,d]); 
    y_label = tf.placeholder(tf.float32,[batch_size,nc]); 

    K0 = 8   
    F0 = 22  
    ncl0 = K0*K0*F0

    K1 = 4   
    F1 = F0  
    ncl1 = K1*K1*F1

    keep_prob_input=tf.placeholder(tf.float32)

    # tahap konvolusi
    # 1
    W_conv1=weight_variable([K0,K0,1,F0])
    b_conv1=bias_variable([F0])
    x_2d1 = tf.reshape(xin, [-1,48,48,1])

    h_conv1=tf.nn.relu(conv2d(x_2d1, W_conv1) + b_conv1)
 
    # 2
    W_conv2=weight_variable([K0,K0,F0,F0])
    b_conv2=bias_variable([F0])

    h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)
    h_conv2_pooled = max_pool_2x2(h_conv2)

    # 3
    W_conv3=weight_variable([K0,K0,F0,F0])
    b_conv3=bias_variable([F1])
    x_2d3 = tf.reshape(h_conv2_pooled, [-1,24,24,F0])

    h_conv3=tf.nn.relu(conv2d(x_2d3, W_conv3) + b_conv3)

    # 4 
    W_conv4=weight_variable([K1,K1,F1,F1])
    b_conv4=bias_variable([F1])

    h_conv4 = tf.nn.relu(conv2d(h_conv3,W_conv4)+b_conv4)
    h_conv4_pooled = max_pool_2x2(h_conv4)
    h_conv4_pooled = max_pool_2x2(h_conv4_pooled)

    # 5
    W_conv5=weight_variable([K1,K1,F1,F1])
    b_conv5=bias_variable([F1])
    x_2d5 = tf.reshape(h_conv4_pooled, [-1,6,6,F1])

    h_conv5=tf.nn.relu(conv2d(x_2d5, W_conv5) + b_conv5)

    # 6
    W_conv6=weight_variable([K1,K1,F1,F1])
    b_conv6=bias_variable([F1])
    b_conv6= tf.nn.dropout(b_conv6,keep_prob_input)

    h_conv6 = tf.nn.relu(conv2d(h_conv5,W_conv6)+b_conv6)
    h_conv6_pooled = max_pool_2x2(h_conv6)

    # membentuk ulang gambar untuk disatukan tiap layer konvolusinya
    h_conv6_pooled_rs = tf.reshape(h_conv6, [batch_size,-1])
    W_norm6 = weight_variable([  6*6*F1, nc])
    b_norm6 = bias_variable([nc])

    # menyatukan setiap layer konvolusi
    h_full6 = tf.matmul( h_conv6_pooled_rs, W_norm6 )
    h_full6 += b_norm6

    y = h_full6; 

    y = tf.nn.softmax(y)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), 1))
    total_loss = cross_entropy

    # optimisasi prediksi
    train_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)

    # akurasi prediksi
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y, xin, keep_prob_input

def plot_face(classified_emotion_vector, face_img):
    emotion_nr = np.argmax(classified_emotion_vector)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    ax1.imshow(np.reshape(face_img, (48,48)))
    ax1.axis('off')
    ax1.set_title(str_emotions[emotion_nr])
    ax2.bar(np.arange(nc) , classified_emotion_vector)
    ax2.set_xticks(np.arange(nc))
    ax2.set_xticklabels(str_emotions, rotation=45)
    ax2.set_yticks([])

    return plt

def preprocess_faces(data_orig):
    n = data_orig.shape[0]
    data = np.zeros([n,48**2])
    for i in range(n):
        xx = data_orig[i,:,:]
        xx -= np.mean(xx)
        xx /= np.linalg.norm(xx)
        data[i,:] = xx.reshape(2304); 
    return data