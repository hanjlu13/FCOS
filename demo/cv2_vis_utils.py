import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont


class Cv2OverlayUtils:
    @staticmethod
    def overlay_text(
        img, text, axis, textColor=(255, 0, 0, 0), textSize=20, convert=True
    ):
        """ overlay text on an ndarray
        
        Args:
            img (numpy.ndarray): original image to be overlayed
            text (list or str): a single string or a list of strings 
            axis (list of list of list): a list of two elements of a list of list   
            textColor (tuple, optional): color of text. Defaults to (255,0,0,0).
            textSize (int, optional): size of textj. Defaults to 20.
            convert (bool, optional): whether to connvert BGR to RGB Defaults to True.
        
        Returns:
            numpy.ndarray: images with text
        """
        # check if the image in numpy ndarray
        assert isinstance(img, numpy.ndarray)

        # chech the type of axis
        # assert isinstance(axis, list)

        # convert text or axis to be element of another list if needed
        if type(text) is str:
            text = [text]
        if not isinstance(axis[0], list):
            axis = [axis]

        # convert BGR into RGB if needed
        if convert:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(img)

        # overlay text
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            "/jp_lab/video_object_detection_on_web_server/utils/simsun.ttc",
            textSize,
            encoding="utf-8",
        )
        for _text, _axis in zip(text, axis):
            draw.text((_axis[0], _axis[1]), _text, textColor, font=fontText)
        return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def overlay_bbox_and_label():
        pass
