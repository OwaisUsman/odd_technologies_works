

# import numpy as np
# import pandas as pd
# df = pd.read_csv('new_sample.csv')
# print (df)
# df
# df.dtypes
# #df.drop(df.columns[5:20], axis=1, inplace=True)
# df.rename(columns={'0': 'data.timestamp'})
# df.head(5)
# df['data.rawTranscriptions'][1]
# df['data.rawTranscriptions'][0]

# import json
# df['data.rawTranscriptions'] = df['data.rawTranscriptions'].apply(lambda x: json.loads(x))
# df

# data=df.explode('data.rawTranscriptions')
# data

# data['data.number']=data['data.number'].astype(int)

# data

# data['data.id']=data['data.id'].astype(int)
# data
# data["audioname"] = data["data.application"].astype(str)+'-' + data["data.id"].astype(str)+'-in.wav'
# data
# data.head(5)

# chunk_name = []
# def chunk(x):
#     for num,val in enumerate(x):
#         chunk_name.append(str(num+1))
# df["data.rawTranscriptions"].apply(chunk)
# chunk_name[0:10]
# df
# data
# data['chunk_name'] = data["data.application"].astype(str)+'-' + data["data.id"].astype(str)+'-in' + chunk_name + '.wav'

# data


# # In[28]:


# get_ipython().system('pip install pydub')


# # In[4]:


# get_ipython().system('pip install librosa')


# # In[6]:


# import matplotlib.pyplot as plt


# # In[7]:


# import librosa
# import librosa.display
# data, sampling_rate = librosa.load("ARI5-1635455564.512-in.wav");
# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(data,sampling_rate)


# # In[48]:


# from pydub import AudioSegment
# sound_file = AudioSegment.from_wav("ARI5-1635455564.512-in.wav")


# # In[66]:


# import IPython.display as ipd
# ipd.Audio('ARI5-1635456032.1751-in.wav')


# # In[61]:


# from pydub import AudioSegment
# from pydub.silence import split_on_silence
 
# sound_file = AudioSegment.from_wav("ARI5-1635455564.512-in.wav")
# audio_chunks = split_on_silence(sound_file, min_silence_len=800, silence_thresh=-30 )
 
# for i, chunk in enumerate(audio_chunks):
#    out_file = "chunk{0}.wav".format(i)
#    print("exporting", out_file)
#    chunk.export(out_file, format="wav")


# # In[33]:


# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# sound_file1 = AudioSegment.from_wav("ARI5-1635455541.404-in.wav")
# audio_chunks = split_on_silence(sound_file1, min_silence_len=800, silence_thresh=-30 )
 
# for i, chunk in enumerate(audio_chunks):
#    out_file = "chunk{0}.wav".format(i)
#    print("exporting", out_file)
#    chunk.export(out_file, format="wav")


# # In[35]:


# sound_file2 = AudioSegment.from_wav("ARI5-1635456032.1751-in.wav")
# audio_chunks = split_on_silence(sound_file2, min_silence_len=800, silence_thresh=-30 )
 
# for i, chunk in enumerate(audio_chunks):
#    out_file = "chunk{0}.wav".format(i)
#    print("exporting", out_file)
#    chunk.export(out_file, format="wav")


# # In[ ]:


''''from pydub import AudioSegment
from pydub.silence import split_on_silence
sound_file = AudioSegment.from_wav("C:/Users/trainee4/Documents/Trainnee44/ARI5-1635780928.18-in.wav")
audio_chunks = split_on_silence(sound_file, min_silence_len=1800, silence_thresh=-48 )
 
for i, chunk in enumerate(audio_chunks):
    print(i)
    out_file = "MySortedData/chunks/ARI5-1635780928.18-in-{0}.wav".format(i)
    print("exporting", out_file)
    chunk.export(out_file, format="wav")'''

import os
from pydub import AudioSegment
from pydub.silence import split_on_silence



# name_list = os.listdir('C:/Users/trainee9/Desktop/Sohaib/T_1') 
source ='//192.168.88.18/oddtech/audios/T_1/'
name_list = os.listdir(source)
count = 67631
main_dir = "C:/Users/trainee7/Desktop/owais usman/newChunk/"

name_list_2 = name_list[67631:]

for name in name_list_2:
    try:
        count = count +1
        sound_file = AudioSegment.from_wav(os.path.join(source ,str(name)))
        audio_chunks = split_on_silence(sound_file, min_silence_len=1900, silence_thresh=-35,  keep_silence=20 )
        name = name.replace('.wav','')
        for i, chunk in enumerate(audio_chunks):
            out_file =os.path.join(main_dir, str(count)+name+"-{0}.wav".format(i+1))
            print("exporting", out_file)
            if not os.path.exists(main_dir):
                os.mkdir(main_dir)
            chunk.export(out_file, format="wav")
        if count%10000==0:
            print("*"*30)
            print(count)
    except Exception as e:
        print(e)

# name_list = os.listdir('Z:/audios/T_1') 
# count = 0
# try:
#     for name in name_list:
#         count = count +1
#         sound_file = AudioSegment.from_wav("Z:/audios/T_1/"+str(name))
#         audio_chunks = split_on_silence(sound_file, min_silence_len=1900, silence_thresh=-35,  keep_silence=20 )
#         name = name.replace('.wav','')
#         for i, chunk in enumerate(audio_chunks):
#             out_file = "C:/Users/trainee7/Desktop/owais usman/aud_chunks/"+name+"-{0}.wav".format(i+1)
#             print("exporting", out_file)
#             chunk.export(out_file, format="wav")
# except (FileNotFoundError):
#     pass
# except (RuntimeError):
#     pass
# except (RuntimeWarning):
#     pass


# from scipy.io.wavfile import read
# samprate, wavdata = read('ARI5-1635455564.512-in.wav')


# In[40]:


# import numpy as np
# chunks = np.array_split(wavdata, numchunks)


# # In[47]:


# import numpy as np
# import math
# from math import sqrt
# import statistics 
# # basically taking a reading every half a second - the size of the data 
# # divided by the sample rate gives us 1 second chunks so I chop 
# # sample rate in half for half second chunks
# chunks = np.array_split(wavdata, wavdata.size/(samprate/2))
# dbs = [20*math.log10( math.sqrt(statistics.mean(chunk**2)) ) for chunk in chunks]
# print(dbs)



# pip install scipy


# # In[52]:


# import IPython.display as ipd
# au=ipd.Audio('ARI5-1635455564.512-in.wav')
# au

# aud_len = len(sound_file)
# aud_len


# import pyglet

# player = pyglet.media.Player()
# sound = pyglet.media.load('ARI5-1635455564.512-in.wav', streaming=False)
# player.queue(sound)
# player.play()


# # In[68]:


# pip install pydub
# pip install noisereduce


# # In[ ]:


# import noisereduce as nr
# # load data
# rate, data = wavfile.read("ARI5-1635455564.512-in.wav")
# # perform noise reduction
# reduced_noise = nr.reduce_noise(y=data, sr=rate)

