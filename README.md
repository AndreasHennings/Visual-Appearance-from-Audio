{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "VAFA.ipynb",
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyMpddbb9J5VnCUyBpjjRFdZ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AndreasHennings/Visual-Appearance-from-Audio/blob/main/VAFA.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2pMR01VtE4jn"
      },
      "source": [
        "# **Visual Appearance from Audio**\n",
        "\n",
        ">Universität Regensburg\n",
        "\n",
        ">Institut für Medieninformatik\n",
        "\n",
        ">Praxisseminar Sommersemester 2020\n",
        "\n",
        ">Dozenten: Prof Wolff, Henze\n",
        "\n",
        ">Abgabedatum: 30.09.2020\n",
        "\n",
        ">Autor: Andreas Hennings\n",
        "\n",
        ">Matr. Nr.: 1755557\n",
        "\n",
        ">Email: Andreas.Hennings@ur.de\n",
        "\n",
        "**Project mission**\n",
        "\n",
        "The main objective of this project is to create a video of a speaking person from an audio file.\n",
        "\n",
        "Secondary objectives are:\n",
        "\n",
        "\n",
        "*   The project can be operated by users with average technical expertise\n",
        "*   The project should deliver results within reasonable time\n",
        "\n",
        "**Prerequesites**\n",
        "\n",
        "* A Google Account and an accesible Google Drive are needed for the project to store uploaded and generated data.\n",
        "\n",
        "* Since the projects calculations are run on external servers, there are almost no hardware requirements for the user.\n",
        "\n",
        "* This project was developed and tested using the Brave browser and should therefore run on Google Chrome as well. Possible issues while using this application may derive from browser incompabilites or add-ons that prevent code-execution (e.g. NoScript) or adblockers.\n",
        "\n",
        "* A source video must be uploaded to provide the algorithm with the necessary data. The video should meet the following prerequesites:\n",
        ">* The video should be in .mp4 format (other formats may work). There is no prerequesite for the videos resolution, but high resolutions will increase the time the application needs to do its calculations.\n",
        ">* Depending on the videos fps, it should be neither too long or to short. It is recommended that there should be several hundred or thousand single images for the algorithm to chose from, so - depending on the source videos fps - a couple of minutes of recording should do. The video should also not be too long, since that will also increase the time the application needs.\n",
        ">*The video should be cropped to the speakers face. There should be a neutral background without moving objects, and the speaker should try to avoid head movements.\n",
        "* An audio track containing the sound for the video to be produced must be uploaded. For best results, this audio file should be recorded using the same microphone that was used for recording the source video file.\n",
        "\n",
        "\n",
        "**Methodology**\n",
        "\n",
        "This project is based on the hypothesis that the sound produced by a speaking person correlates to his/her facial expressions, especially lip movement. This hypothesis is in turn based on the observation that e.g. people with hearing-impairment can infer the sound a speaking person produces by observing their mouth (\"lip-reading\").\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "To achieve the desired results, this project combines a number of data-processing technologies. In short:\n",
        "\n",
        "* First, this application extracts audio data from a source video file, calculates the distinct vocal features and correlates this data to the corresponding images extracted from the source video using a mean-shift algorithm for hierarchical clustering of multi-dimensional data.\n",
        "\n",
        "* Next, the application extracts those features from a new audio file and determines which of the stored images of the source video correlate to those features. From this list of images, a image-similarity algorithm then chooses a single image that fits best.\n",
        "\n",
        "* Finally, all images are combined into a video file and the new Audio is added.\n",
        "\n",
        "Note: While implementing this projects, some code was written that is not necessary for the application to work, but shows intermediary results or visualizes data structures. I deliberately chose to not delete these cells, since they can help to better understand the methodology, find possible issues or tweak parameters for best results. These cells are marked as '(Optional)' and do not have to be executed.\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oOz1umpYUI9L"
      },
      "source": [
        "# Data Preprocessing"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-P8VSjjtQih4"
      },
      "source": [
        "**Mount drive**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RI0ZDb5UKyoT",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "5008fd92-ccbb-4f92-bf56-b84313790e90"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/gdrive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TZCr_Y3FVqgI"
      },
      "source": [
        "**Get Source videos framerate**\n",
        "\n",
        "This is necessary to determine in how many packets the audio data needs to be split"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JEKmzU4enHZq"
      },
      "source": [
        "#Get Framerate as expression\n",
        "frameRate = !ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 \"/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/V2_src_c.m4v\"\n",
        "\n",
        "#Calculate and round Framerate\n",
        "fps = round(float(eval(str(frameRate[0]))),2)\n",
        "\n",
        "print(\"Your framerate is \"+str(fps)+ \" fps\")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_mqSAcyYcr3W"
      },
      "source": [
        "**Extract Audio from Source Video and save it on Google Drive**\n",
        "\n",
        "Note: if the audio file already exists, you are asked to confirm overwriting it at the bottom of the cell below"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LgyTqaLWcwv8"
      },
      "source": [
        "!ffmpeg -i \"/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/V2_src_c.m4v\" -vn \"/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/V2_audio.mp3\""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9AFMbE4eQa6o"
      },
      "source": [
        "**Open Audio for MFCC calculation**\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "l4QVdWmkMJb1"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import librosa, librosa.display\n",
        "import numpy as np\n",
        "\n",
        "srcAudio = \"/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/V2_audio.mp3\"\n",
        "newAudio = \"/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/V2_audio_new.mp3\"\n",
        "\n",
        "\n",
        "\n",
        "srcSignal, sr = librosa.load(srcAudio, sr=22050) # sr = samplerate\n",
        "newSignal, sr = librosa.load(newAudio, sr=22050)\n",
        "\n",
        "# n_fft is the number of samples to be used for calculation\n",
        "# hop_length is the number of samples each step is apart\n",
        "\n",
        "n_fft = round(22050/fps)\n",
        "hop_length = n_fft\n",
        "\n",
        "print (n_fft)\n",
        "print (hop_length)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dweea6LOQStF"
      },
      "source": [
        "**(Optional) Display audio features**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r2sU6QS-JZgk"
      },
      "source": [
        "# Show waveform\n",
        "print(\"Waveform\")\n",
        "librosa.display.waveplot(srcSignal, sr=sr)\n",
        "plt.xlabel(\"Time\")\n",
        "plt.ylabel(\"Amplitude\")\n",
        "plt.show()\n",
        "\n",
        "# Show power spectrum using Fourier Transformation\n",
        "print(\"Powerspectrum\")\n",
        "fft = np.fft.fft(srcSignal)\n",
        "mag = np.abs(fft)\n",
        "freq = np.linspace(0, sr, len(mag))\n",
        "l_freq = freq[:int(len(freq)/2)]\n",
        "l_mag = mag[:int(len(mag)/2)]\n",
        "#show graph\n",
        "plt.plot(l_freq, l_mag)\n",
        "plt.xlabel(\"Freq\")\n",
        "plt.ylabel(\"Magnitude\")\n",
        "plt.show()\n",
        "\n",
        "#Show spectrogram\n",
        "print(\"Spectrogram\")\n",
        "stft = librosa.core.stft(srcSignal, hop_length=hop_length, n_fft=n_fft)\n",
        "spec = np.abs(stft)\n",
        "log_spec = librosa.amplitude_to_db(spec)\n",
        "librosa.display.specshow(log_spec, sr=sr, hop_length=hop_length)\n",
        "plt.xlabel(\"Time\")\n",
        "plt.ylabel(\"Frequency\")\n",
        "plt.colorbar()\n",
        "plt.show()\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mu12cYLQbIox"
      },
      "source": [
        "**Calculate MFCCs**\n",
        "\n",
        "The raw audio data stored in the file is not suitable for determining which sound has been produced by the speaker, so we need to calculate the Mel-frequency cepstral coefficients (MFCCs).\n",
        "\n",
        "\n",
        "n_mfcc is the number of MFCCs to be calculated for each frame and can be tweaked to improve results.\n",
        "\n",
        "Since the results are in a list ordered by MFCCs, the resulting matrix m needs to be transposed to create a list ordered by framesnumbers.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zxXmSQCebL_t"
      },
      "source": [
        "srcMFCCs = librosa.feature.mfcc(srcSignal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)\n",
        "srcM = srcMFCCs.transpose()\n",
        "print(len(srcM))\n",
        "\n",
        "newMFCCs = librosa.feature.mfcc(newSignal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)\n",
        "newM = newMFCCs.transpose()\n",
        "print(len(newM))\n",
        "\n",
        "m = np.concatenate((srcM,newM))\n",
        "print (len(m))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mq2ahGGFtlcr"
      },
      "source": [
        "**(Optional) Show MFCC spectrogram and array dimensions**\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "g-LnOulLuKsp"
      },
      "source": [
        "#show spectrogram\n",
        "\n",
        "librosa.display.specshow(m, sr=sr, hop_length=hop_length)\n",
        "plt.xlabel(\"Time\")\n",
        "plt.ylabel(\"MFCC\")\n",
        "plt.colorbar()\n",
        "plt.show()\n",
        "\n",
        "print (len(m))\n",
        "print (len(m[0]))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_EIbG_7fAymb"
      },
      "source": [
        "**Importing video and converting to image sequence**\n",
        "\n",
        "We import the video and store the images. Also, we convert the images to greyscale and store it in an other array."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kpuXZegRbJZi"
      },
      "source": [
        "import cv2\n",
        "from skimage.transform import resize\n",
        "\n",
        "images=[]\n",
        "gImages=[]\n",
        "\n",
        "vidcap = cv2.VideoCapture(\"/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/V2c.m4v\")\n",
        "success,image = vidcap.read()\n",
        "count = 0\n",
        "while success:\n",
        "  success,image = vidcap.read()\n",
        "  print('Reading frame: ', count)\n",
        "  count += 1\n",
        "  images.append(image)\n",
        "  try:\n",
        "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
        "    g = resize(gray,(256,256))\n",
        "    gImages.append(g)\n",
        "  except: #If we cant create a bw-image, erase src-image from stack\n",
        "    images.pop()\n",
        "    print(\"something went wrong\")\n",
        "\n",
        "  \n",
        " \n",
        "  \n",
        "print (len(images))\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Q-OAOgIX14LS"
      },
      "source": [
        "print (images[0].shape)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YHbK2ei_4MfC"
      },
      "source": [
        "# Video Synthesis\n",
        "\n",
        "With the MFCCs calculated from audio data, we can now implement different solutions for video syntheses "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qgLf0qXt5G3u"
      },
      "source": [
        "**Solution 1: Calculating the Euclidian Distance**\n",
        "\n",
        "The MFCCs represents n Data Points in a k-Dimensional Space, where \n",
        "* n = number of frames \n",
        "* k = number of MFCCs\n",
        "\n",
        "Understanding MFCCs as vectors, for each MFCC in our new Audio we can find the closest (= most similar) MFCC in our source audio.\n",
        "\n",
        "Then, we can select the corresponding image and add it to the video."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZDalAtkT5pw9"
      },
      "source": [
        "\n",
        "height, width, layers = images[0].shape\n",
        "size = (width,height)\n",
        "video = cv2.VideoWriter('/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/result_euclidian.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)\n",
        "\n",
        "for n in range(len(newM)):  \n",
        "  nm = newM[n]\n",
        "  minDist = 1000.0\n",
        "\n",
        "  for s in range(len(srcM)):\n",
        "    sm = srcM[s]\n",
        "    dist = np.linalg.norm(nm-sm)\n",
        "    \n",
        "    if (dist < minDist):\n",
        "      minDist = dist\n",
        "      numImg = s\n",
        "\n",
        "  imgList.append(images[numImg])  \n",
        "  video.write(images[numImg])\n",
        "\n",
        "video.release()\n",
        "  \n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yVOfXMOW5qnD"
      },
      "source": [
        "**Solution 2: Flat clustering using K-Means**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rRPKvcks0zue"
      },
      "source": [
        "from sklearn.cluster import KMeans\n",
        "\n",
        "n_clusters = 64\n",
        "kmeans = KMeans(n_clusters=n_clusters)\n",
        "kmeans.fit(m)\n",
        "\n",
        "clusters = kmeans.predict(m)\n",
        "srcClusters = clusters[:len(srcM)]\n",
        "newClusters = clusters[len(srcM):]\n",
        "\n",
        "for c in srcClusters:\n",
        "  print (c)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ifGc1dgid4up"
      },
      "source": [
        "**Create a look-up-table** \n",
        "\n",
        "We need to create a list, where for each cluster ID all coresponding framenumbers are stored.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vFvnpl2Bd6Fa"
      },
      "source": [
        "\n",
        "labels_unique = np.unique(srcClusters)\n",
        "n_clusters_ = len(labels_unique)\n",
        "\n",
        "print(\"Number of unique labels: \",n_clusters_)\n",
        "print()\n",
        "\n",
        "lut=[] #creating the look-up table\n",
        "\n",
        "for n in range(n_clusters): #0-64\n",
        "  arr = np.where (srcClusters==n)\n",
        "  arr = list(arr[0])\n",
        "  \n",
        "  lut.append(arr)  \n",
        "  print(\"Cluster ID: %d in frames:\" %n)\n",
        "  print(arr)\n",
        "  \n",
        "\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ltF4zwtOrk_f"
      },
      "source": [
        "**Calculate Image Similarity**\n",
        "\n",
        "Since we have a list of frames for each cluster, we can determine the best image to append to our result video. To avoid flickering, we choose the most similar image.\n",
        "\n",
        "Depending on the number of frames to choose from, this may take several seconds for each frame. \n",
        "\n",
        "I therefore added a threshold that can optionally by used to shorten calculation time. The value of 0.95 was empirically determined."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GcuD5HukvwEK"
      },
      "source": [
        "from skimage.measure import compare_ssim\n",
        "import imutils\n",
        "\n",
        "\n",
        "def ImageSimilarity(img, imgs):\n",
        "  threshold = 0.95\n",
        "  resultImgNr = 0\n",
        "  maxSim = 0.0\n",
        "\n",
        "  \n",
        "  for i in imgs:\n",
        "\n",
        "    if i != img:\n",
        "      (score, diff) = compare_ssim(gImages[img], gImages[i], full=True)\n",
        "      diff = (diff * 255).astype(\"uint8\")\n",
        "\n",
        "      #if score>threshold:\n",
        "        #return i\n",
        "      \n",
        "      if score > maxSim:\n",
        "        maxSim = score\n",
        "        resultImgNr = i\n",
        "  \n",
        "  print (\"resultImgNr: \", resultImgNr)\n",
        "  return resultImgNr\n",
        "  "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nhAcBtZl6Ooz"
      },
      "source": [
        "**Image Selection**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yDuPBrkrqMZo"
      },
      "source": [
        "img0 = 0\n",
        "counter = 0\n",
        "resultImages=[]\n",
        "\n",
        "for nc in newClusters:\n",
        "\n",
        "  \n",
        "  while (len(lut[nc])==0):\n",
        "    nc -= 1\n",
        "  print(\"counter: \",counter,\" of \", len(newClusters),\"clusterNr: \",nc,\"Nr of entries:\",len(lut[nc]), \"Images\", lut[nc])\n",
        " \n",
        "  img1 = ImageSimilarity(img0,lut[nc])  \n",
        "  resultImages.append(img1)  \n",
        "    \n",
        "      \n",
        "  counter +=1\n",
        "  img0 = img1\n",
        "#print (resultImages)\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rh6i9brEUkr6"
      },
      "source": [
        "**Video Synthesis**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "61fpXJlnUdf3"
      },
      "source": [
        "height, width, layers = images[0].shape\n",
        "size = (width,height)\n",
        "video = cv2.VideoWriter('/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/result_v2_meanshift100.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)\n",
        "\n",
        "for n in range(0,len(resultImages),4):\n",
        "  nr = resultImages[n]\n",
        "  video.write(images[nr])\n",
        "  video.write(images[nr+1])\n",
        "  video.write(images[nr+2])\n",
        "  try:\n",
        "    nn = resultImages[n+1]\n",
        "  except:\n",
        "    nn = nr\n",
        "  a=images[nr+2]\n",
        "  b=images[nn]\n",
        "  \n",
        "  inter = cv2.addWeighted(a, 0.5, b, 0.5, 0)\n",
        "\n",
        "  video.write(inter)\n",
        "  \n",
        "video.release()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YhABQ9jV6uV0"
      },
      "source": [
        "**Add Sound to Video**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "65MqyG55jnM_"
      },
      "source": [
        "!ffmpeg -i '/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/result_v2_meanshift100.mp4' -i \"/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/V2_audio_new.mp3\" -map 0:v -map 1:a -c:v copy -shortest '/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/result_v2_meanshift100_audio.mp4'"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Tgvw4IHp62KL"
      },
      "source": [
        "**(Optional) Interpolate Images**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BWth0fUf21aw"
      },
      "source": [
        "!ffmpeg -i '/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/result_v2_meanshift100_audio.mp4' -filter:v minterpolate -r 120 '/content/gdrive/My Drive/Visual Appearance from Audio/ExampleProject/result_v2_meanshift100_audio_inter.mp4'"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9ey2jQPuEWKT"
      },
      "source": [
        "**Solution 3: Hierachical Clustering using MeanShift**\n",
        "\n",
        "The MFCCs represents n Data Points in a k-Dimensional Space, where \n",
        "* n = number of frames \n",
        "* k = number of MFCCs\n",
        "\n",
        "To determine the individually distinctable sounds these data points represent, we need to find clusters of neighboring data points.\n",
        "\n",
        "We're using the MeanShift algorithm to determine where our datapoints cluster and calculate to which cluster our MFCCs correlate.\n",
        "\n",
        "The parameter 'quantile' determines the number of clusters that are calculated and can be tweaked to improve results. The smaller the value, the more clusters are detected. \n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yDtac8KKEtEW"
      },
      "source": [
        "from sklearn.cluster import MeanShift, estimate_bandwidth\n",
        "\n",
        "# The following bandwidth can be automatically detected using\n",
        "bandwidth = estimate_bandwidth(m, quantile=0.01, n_samples=3000)\n",
        "\n",
        "# Calculate Clusters\n",
        "ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)\n",
        "ms.fit(m)\n",
        "\n",
        "allClusters = ms.labels_\n",
        "\n",
        "srcClusters = allClusters[:len(srcM)]\n",
        "newClusters = allClusters[len(srcM):]\n",
        "\n",
        "\n",
        "mlabels_unique = np.unique(allClusters)\n",
        "mn_clusters_ = len(mlabels_unique)\n",
        "\n",
        "print(\"Nr of Clusters:\", mn_clusters_,\" src:\",len(srcClusters),\" new:\",len(newClusters))\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pSfRV_217DPU"
      },
      "source": [
        "**Create Look-up-table**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Go2GXe5spLfb"
      },
      "source": [
        "lut=[]\n",
        "for n in range(mn_clusters_): #0-64\n",
        "  arr = np.where (srcClusters==n)\n",
        "  arr = list(arr[0])\n",
        "  a = arr[:100]\n",
        "\n",
        "  lut.append(a)  \n",
        "  print(\"Cluster ID: %d in frames:\" %n,\" len: \",len(a))\n",
        "  print(a)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MgWcVAUC5so9"
      },
      "source": [
        "**Calculating Video from Selected Frames**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AF9fotC9g3pv"
      },
      "source": [
        "img0 = 0\n",
        "counter = 0\n",
        "resultImages=[]\n",
        "\n",
        "for nc in newClusters:\n",
        "\n",
        "  \n",
        "  while (len(lut[nc])==0):\n",
        "    nc -= 1\n",
        "  print(\"counter: \",counter,\" of \", len(newClusters),\"clusterNr: \",nc,\"Nr of entries:\",len(lut[nc]), \"Images\", lut[nc])\n",
        "  print()\n",
        "  img1 = ImageSimilarity(img0,lut[nc])  \n",
        "  print (\"Result:\", img1)\n",
        "  resultImages.append(img1)  \n",
        "    \n",
        "      \n",
        "  counter +=1\n",
        "  img0 = img1\n",
        " \n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
