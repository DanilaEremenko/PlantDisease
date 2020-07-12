import pandas as pd
import sys
import io

if len(sys.argv) != 2:
    raise Exception("argc !=2, exiting...")

file_text = ''
with open(sys.argv[1], 'r') as fp:
    for line in fp.readlines():
        file_text += "%s\n" % line

file_text = file_text.replace(',,,\n', '')

df = pd.read_csv(io.BytesIO(file_text.encode()), dtype='unicode', sep=',')

for i, row in enumerate(df.sort_values(by=['Авторы']).iterrows()):
    row = row[1]
    if str(row['Парсить']) != 'nan':
        if row['Авторы'] != 'NO' and str(row['Авторы']) != 'nan':
            print("%d. %s. %s [электронный ресурс]:%s" % (i+1, row['Авторы'], row['Название'], row['Ссылка']))
        else:
            print("%d. %s [электронный ресурс]:%s" % (i+1, row['Название'], row['Ссылка']))
