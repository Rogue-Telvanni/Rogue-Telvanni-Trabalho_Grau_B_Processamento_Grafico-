import cv2 as cv
from tkinter import colorchooser
from pathlib import Path
from datetime import datetime

from Image_Filtering import Image_Filtering
from Render_Type import Render_Type

filtering = Image_Filtering()
projectname = "Trabalho Grau B"
filter_applied = 0


# TODO: Add color filter in the image | ok
# TODO: Add a color selector form 0 to 255 RGB color | ok
# TODO: Add Negative to images | ok
# TODO: Be able to remove sticker using the R key | ok
# TODO: Be able to use face detection while in the camera renderer | ok
# TODO: Add Stickers using face position data | kinda


def main():
    # add support to change from image to camera
    stop = False
    render_type = Render_Type.IMAGE
    # training data get from the open vc project to find faces at
    # https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # training data get from the open vc project to find eyes at
    # https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

    while not stop:
        render_type, stop = render(render_type, eye_cascade, face_cascade)


def render(render_type, eye_cascade, face_cascade):
    filtering.reset()
    has_color_filter = False
    color_filter = None
    is_negative = False
    # get camera data and add a filter and show the image
    cv.namedWindow(projectname, cv.WINDOW_GUI_EXPANDED)
    cv.setWindowProperty(projectname, cv.WND_PROP_TOPMOST, cv.WINDOW_FULLSCREEN)
    cv.createTrackbar("filter", projectname, 0, 7, on_change)
    cv.createTrackbar("sticker", projectname, 0, 4, on_change_sticker)

    if render_type == Render_Type.CAMERA_VIDEO:
        draw_face = True
        capture = cv.VideoCapture(0)
        if not capture.isOpened():
            print("Erro ao capturar camera, voltando para uso da imagem")
            return Render_Type.IMAGE, False

        while True:
            ret, frame = capture.read()
            if frame is None:
                break

            frame = cv.flip(frame, 1)
            eye_x, eye_y = get_face_position(eye_cascade, face_cascade, frame, draw_face)
            if len(filtering.stickers_position) > 0:
                filtering.stickers_position.pop(0)
            elif cv.getTrackbarPos("sticker", projectname) > 0 and eye_x > 0:
                filtering.stickers_position.append((1, eye_x, eye_y))

            frame = filtering.add_image_overlays(frame)
            frame = filtering.add_filter(frame, is_negative, has_color_filter, color_filter)
            # convert to gray scale of each frames
            write_legend(frame, True)
            cv.imshow(projectname, frame)
            digit = cv.pollKey()
            # letra i
            if digit == 105:
                cv.destroyAllWindows()
                return Render_Type.IMAGE, False
            # letra G
            elif digit == 103:
                cv.imwrite(f"Imagem_{datetime.now()}", frame)
            elif digit == 27:
                return render_type, True
            elif digit == 110:
                is_negative = not is_negative
            elif digit == 112:
                color_filter = get_user_selected_color()
            # letra S
            elif digit == 115:
                draw_face = not draw_face
            # letra F
            elif digit == 102:
                has_color_filter = not has_color_filter
            # letra R
            elif digit == 114:
                if len(filtering.stickers_position) > 0:
                    filtering.stickers_position.pop(len(filtering.stickers_position) - 1)
    else:
        cv.setMouseCallback(projectname, mouse_click)
        image_path = 'images/lena.png'
        while True:
            image = cv.imread(image_path)
            image = cv.resize(image, (650, 500))
            filtered_image = filtering.add_image_overlays(image)
            filtered_image = filtering.add_filter(filtered_image, is_negative, has_color_filter, color_filter)
            write_legend(filtered_image)
            cv.imshow(projectname, filtered_image)
            digit = cv.pollKey()
            print(digit)
            # letra V
            if digit == 118:
                cv.destroyAllWindows()
                return Render_Type.CAMERA_VIDEO, False
            # letra G
            elif digit == 103:
                cv.imwrite(f"Imagem_{datetime.now()}.png", filtered_image)
            # ESC
            elif digit == 27:
                return render_type, True
            # letra N
            elif digit == 110:
                is_negative = not is_negative
            # letra P
            elif digit == 112:
                color_filter = get_user_selected_color()
            # letra F
            elif digit == 102:
                has_color_filter = not has_color_filter
            # tecla 1
            elif digit == 49:
                image_path = get_next(image_path, 1)
            # tecla 2
            elif digit == 50:
                image_path = get_next(image_path, 0)
            # letra R
            elif digit == 114:
                if len(filtering.stickers_position) > 0:
                    filtering.stickers_position.pop(len(filtering.stickers_position) - 1)


def get_next(image_path, direction):
    image_list = []
    for child in Path('images').iterdir():
        if child.is_file():
            image_list.append(f"{child.parent}/{child.name}")
    index = image_list.index(image_path, 0, len(image_list))
    if direction == 0:
        if index == len(image_list) - 1:
            return image_list[0]
        else:
            return image_list[index + 1]
    elif direction == 1:
        if index == 0:
            return image_list[len(image_list) - 1]
        else:
            return image_list[index - 1]
    else:
        return image_path


# return the left eye y and x position
def get_face_position(eye_cascade, face_cascade, frame, draw_face):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        if draw_face:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # To draw a rectangle in eyes
        if draw_face:
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

        # To draw a rectangle in eyes
        if len(eyes) > 0:
            eye_x, eye_y, eye_w, eye_h = eyes[0]
            return eye_x + eye_w, eye_y + eye_h

    return 0, 0


# Adiciona legenda para os comandos programados
def write_legend(image, is_camera=False):
    cv.putText(image, "P - Color Picker", (0, 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)
    cv.putText(image, "N - Filtro Negativo", (0, 25), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)
    cv.putText(image, "F - Adicionar filtro de cor", (0, 40), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)
    cv.putText(image, "R - Remover sticker", (0, 55), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)
    cv.putText(image, "G - Gravar Imagem", (0, 70), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)
    if is_camera:
        cv.putText(image, "I - Mudar para imagem", (0, 85), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)
        cv.putText(image, "S - \"Stop face Detection\"", (0, 100), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)
    else:
        cv.putText(image, "V - Mudar para camera", (0, 85), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_8)


def get_user_selected_color():
    color_code = colorchooser.askcolor(title="Choose color")
    return color_code


def on_change(value):
    filtering.set_filter_type(value)


def on_change_sticker(value):
    filtering.set_foreground_type(value)


def mouse_click(event, x, y, flags, param):
    # to check if left mouse
    # button was clicked
    if event == cv.EVENT_LBUTTONDOWN:
        # was clicked.
        # get the mouse click position
        # and pass it to the filtering image class
        # if y <= preview_height + 10:

        sticker = cv.getTrackbarPos("sticker", projectname)
        filtering.stickers_position.append((sticker, x, y))


if __name__ == '__main__':
    print(cv.__version__)
    main()
    cv.destroyAllWindows()
