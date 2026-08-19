

import numpy as np 

from torchvision.utils import save_image
from scipy.signal import convolve
from lib.motionblur import Kernel
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

def normalize(x):return (x - x.min()) / (x.max() - x.min())


class Picklelabel_Kernel(Kernel):
    
    def _createKernel(self, save_to: Path=None, show: bool=False):
        if self.kernel_is_generated:
            return None
        # get the path
        self._createPath()

        # (pillow Image object)
        self.kernel_image = Image.new("RGB", self.SIZEx2)

        # make painter local 
        painter = ImageDraw.Draw(self.kernel_image)
        painter.line(xy=self.path, width=int(self.DIAGONAL / 150))

        # applying gaussian blur for realism
        self.kernel_image = self.kernel_image.filter( ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01)))
        self.kernel_image = self.kernel_image.resize( self.SIZE, resample=Image.LANCZOS)
        self.kernel_image = self.kernel_image.convert("L")

        # flag that we have generated a kernel
        self.kernel_is_generated = True

    
    
class Motion_Blur_Generator(Kernel):
    def __init__(self, size: tuple = (100, 100), intensity: float=0, back_to_pil=True ):
        super().__init__(size=size, intensity=intensity )
        self.DIAGONAL = self.DIAGONAL * 0.2
        self.back_to_pil = back_to_pil
        
    def __call__(self, image, conv_mode = "same", ):
        # image = image.convert(mode="RGB")
        self.kernel_is_generated = False 
        self._createKernel()
        # image.save("temp_OG.png")
        result_bands = ()
        for band in image.split():
            # convolve each band individually with kernel
            result_band = convolve(band, self.kernelMatrix, mode=conv_mode).astype("uint8")
            # collect bands
            result_bands += result_band,
        # stack bands back together
        result = np.dstack(result_bands)

        if self.back_to_pil:
            image = Image.fromarray(result)
        else:
            image = result
        # Image.fromarray(result).save("temp.png")
        # image.save("temp.png")
        # quit()

        # Get image
        return image

    



class Pickable_Motion_Blur_Generator(Picklelabel_Kernel):
    def __init__(self, size: tuple = (100, 100), intensity: float=0, back_to_pil=True ):
        super().__init__(size=size, intensity=intensity )
        self.DIAGONAL = self.DIAGONAL * 0.2
        self.back_to_pil = back_to_pil
        
    def __call__(self, image, conv_mode = "same", ):
        
        self.kernel_is_generated = False 
        self._createKernel()
        
        result_bands = ()
        for band in image.split():
            result_band = convolve(band, self.kernelMatrix, mode=conv_mode).astype("uint8")
            result_bands += result_band,
        
        result = np.dstack(result_bands)

        if self.back_to_pil:
            image = Image.fromarray(result)
        else:
            image = result
        # Image.fromarray(result).save("temp.png")
        # image.save("temp.png")
        # quit()

        # Get image
        return image

    
    def __getstate__(self):
        state = self.__dict__.copy()
        # state['kernel_image'].save("temp1.png"), self.kernel_image.save("temp2.png")
        # quit()
        # print(" ==== ", state, self.kernel_image.size )
        if "kernel_image" in state:
            state['kernel_image'] = self.kernel_image.tobytes()
        return state

    def __setstate__(self, state):
        # print(" ==== ", state)
        # Restore the image from the pickled format
        self.__dict__.update(state)
        if "kernel_image" in state:
            self.kernel_image = Image.frombytes('L', self.SIZE, self.kernel_image)