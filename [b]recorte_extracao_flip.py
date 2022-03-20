"""Este código elaborado em estudos realizados em cima de exemplos que podem ser encontrados no
no site <https://www.programcreek.com/python/?CodeExample=detect+faces>"""

from PIL import Image
from mtcnn import MTCNN
from os import listdir
from os.path import isdir
from numpy import asarray

face_detection = MTCNN()


def extrair_face(arquivo, size=(255, 255)):  # (160, 160))
    photo = Image.open(arquivo)
    photo = photo.convert('RGB')
    array = asarray(photo)
    resultado = face_detection.detect_faces(array)

    x1, y1, width, height = resultado[0]['box']
    x2 = x1 + width
    y2 = y1 + height

    face = array[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(size)

    return image


def carrega_fotos(directory_src, directory_target):
    for filename in listdir(directory_src):

        path = directory_src + filename

        path_target = directory_target + filename
        path_target_flip = directory_target + "flip_" + filename
        try:
            face = extrair_face(path)
            flip = flip_image(face)
            face.save(path_target, "JPEG", quality=100, optimize=True,
                      progressive=True)
            flip.save(path_target_flip, "JPEG", quality=100, optimize=True,
                      progressive=True)
        except:
            print(f"Erro de processamento na imagem {path}")


def carrega_dir(directory_src, directory_target):
    for subdir in listdir(directory_src):
        path = directory_src + subdir + "\\"
        path_target = directory_target + subdir + "\\"

        if not isdir(path):
            continue

        carrega_fotos(path, path_target)


def flip_image(image):
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# observação:
# 1º - antes de executar o código é necessário que se crie um diretório de origem com as fotos;
# 2º - criar diretório de destino, onde as imagens recortadas pelo algoritmo serão salvas;
# 3º - os dois diretórios deveram ter suas respectivas subpastas com os nomes de cada pessoa;
# Ex: C:\\dataset\\aula\\fotos\\ "Maria" , "João" ...
#    C:\\dataset\\aula\\faces\\ "Maria", "João"  ...

if __name__ == '__main__':
    # Exemplificação
    # carrega_dir("C:\\dataset\\aula\\fotos\\",   #diretório de origem das imagens/fotos 
    #        "C:\\dataset\\aula\\faces\\")     #diretorio de destino para as imagens/fotos

    carrega_dir("C:\\dataset\\aula\\fotos\\",
             "C:\\dataset\\aula\\faces\\")
