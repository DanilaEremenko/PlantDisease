
from PIL import Image
from PyQt5.QtGui import QImage,QPixmap,QImageWriter
from PyQt5.QtCore import QByteArray, QBuffer, QThread, pyqtSignal
from io import BytesIO


def make_jpeg_matrix(img_massive):
    name = create_bin_jpeg(img_massive)
    return read_bin_jpeg(name)

def create_bin_jpeg(img_massive):
    saving_jpeg_zoom(img_massive, 1)



def saving_jpeg_zoom(img_massive,zoom):
    name = "output\jpeg_array.bin"
    updt_file = open(name, 'wb')
    updt_file.close()

    # img1 = QImage()
    # img1.loadFromData(img_massive[0])
    # imagefile = QImageWriter()
    # imagefile.setFileName("C:\PyProjects\PlantDisease\supertest")
    # imagefile.setQuality(100)
    # print('save ',imagefile.write(img1))
    # print('saving = ',img1.save("C:\PyProjects\PlantDisease\output\one","PNG",-1))
    # print(img1.save("C:\PyProjects\PlantDisease\output\one_zip.jpg",format = 'jpeg'))

    for i in img_massive:
        img = Image.fromarray(i, 'RGB')
        img_ram = BytesIO()
        # img.scaled(img.width * zoom, img.height * zoom)
        img.save(img_ram, format='JPEG', quality=80, optimize=True, progressive=True)
        img_byte_array = img_ram.getvalue()
        updt_file = open(name, 'ab')
        updt_file.write(bytearray(img_byte_array))
        updt_file.close()
    return name


def read_bin_jpeg(name):
    header = b'\xff\xd8'
    tail = b'\xff\xd9'
    updt_file = open(name, 'rb')
    massive = updt_file.read()

    starts = 0
    ends = 0
    delta = 0
    jpeg_mass = []
    while delta < len(massive):
        start = massive.find(header, delta)
        starts += 1
        end = massive.find(tail, delta) + 2
        ends += 1
        delta = end
        jpeg_mass.append(massive[start:end])  # add image container
        # print(start, end, delta)
    print('Start: %d End: %d' % (starts, ends))
    return jpeg_mass






