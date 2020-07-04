from pd_lib.coor_extruder import ImageMetaData
import numpy as np
import math
import os


class GetMozaicMatrix():
    def __init__(self):
        self.photo_width = None
        self.photo_height = None
        super(GetMozaicMatrix, self).__init__()

    def get_matrix(self, photo_path):
        disperse = 0
        min_point = 0
        max_point = 0
        matrix = []
        data_x = []
        data_y = []
        delay = 3
        mat_line = []
        for root, dirs, files in os.walk(photo_path, topdown=True):
            dirs.clear()  # with topdown true, this will prevent walk from going into subs
            images = [i for i in files if '.JPG' in i]
            print(files)
            print(images)
            if len(images) == 0:
                return None, 0

            for image in images:
                meta_data = ImageMetaData(photo_path + '/' + image)
                lat, lng = meta_data.get_lat_lng()
                self.photo_width, self.photo_height = meta_data.get_resolution()

                iterator = len(mat_line) if len(matrix) % 2 == 0 else - len(mat_line)
                if not data_x:
                    data_x.append(lat)
                    data_y.append(lng)
                else:
                    if not lat == data_x[-1] and not lng == data_y[-1]:
                        data_x.append(lat)
                        data_y.append(lng)
                        if len(data_x) >= 3:
                            i = len(data_x) - 1
                            Cx = data_x[i]
                            Cy = data_y[i]
                            Bx = data_x[i - 1]
                            By = data_y[i - 1]
                            Ax = data_x[i - 2]
                            Ay = data_y[i - 2]
                            ABx = (Bx - Ax)
                            ABy = (By - Ay)
                            ACx = (Cx - Ax)
                            ACy = (Cy - Ay)
                            FiTop = ABx * ACx + ABy * ACy
                            FIBot = math.sqrt(ABx * ABx + ABy * ABy) * math.sqrt(ACx * ACx + ACy * ACy)
                            Fi = FiTop / FIBot
                            if delay < 0:
                                if math.degrees(math.acos(Fi)) > 20:
                                    delay = 3
                                    disperse += iterator
                                    if min_point > disperse:
                                        min_point = disperse
                                    if max_point < disperse:
                                        max_point = disperse
                                    matrix.append(mat_line.copy())
                                    mat_line.clear()
                            delay -= 1
                mat_line.insert(iterator, root + '/' + image)
                if image == files[-1]:
                    matrix.append(mat_line)

        imgs_in_line = max_point - min_point
        offset = 0

        len_prev_line = len(matrix[0])
        for line in matrix:
            added = 0
            if (matrix.index(line)) % 2:
                offset += abs(len_prev_line - len(line))

            for y in range(min_point + offset, 0):
                line.insert(0, None)
                added += 1
            len_prev_line = len(line) - added

            if len(line) < imgs_in_line:
                for y in range(imgs_in_line - len(line)):
                    iterator = len(line) if len(matrix) % 2 == 1 else - len(line)
                    line.insert(iterator, None)
        mat_final = np.fliplr(matrix)

        # for line in mat_final:
        #     print('', end='\n')
        #     for img in line:
        #         print(img, end=' ')
        # print('')
        # print('lines ', min_point, max_point)
        # print(mat_final, ' ', len(files))
        return mat_final.transpose(), len(files)

    def get_resolution(self):
        return self.photo_width, self.photo_height


GetMozaicMatrix()
