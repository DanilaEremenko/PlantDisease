from PIL import Image
from io import BytesIO

from PyQt5.QtGui import QImage


def create_bin_jpeg(th, img_massive, zoom_mass):
    for z in zoom_mass:
        saving_jpeg_zoom(th, img_massive, z)


def saving_jpeg_zoom(th, img_massive, zoom):
    name = "output\jpeg_array_" + str(zoom) + ".bin"
    updt_file = open(name, 'wb')
    updt_file.close()
    for i in enumerate(img_massive):
        img = Image.fromarray(i[1], 'RGB')
        img_ram = BytesIO()
        small = img.resize((int(img.width * zoom), int(img.height * zoom)), Image.ANTIALIAS)
        small.save(img_ram, format='JPEG', quality=80, optimize=True, progressive=True)
        img_byte_array = img_ram.getvalue()
        th.progress_signal.emit(int(i[0] * 100 / len(img_massive)))
        updt_file = open(name, 'ab')
        updt_file.write(bytearray(img_byte_array))
        updt_file.close()
    return name


def convert_to_image(massive):
    img = QImage()
    img.loadFromData(massive)
    return img


def read_bin_jpeg(names):
    header = b'\xff\xd8'
    tail = b'\xff\xd9'
    file_mass = []
    for name in names:
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
        file_mass.append(jpeg_mass.copy())
    return file_mass
