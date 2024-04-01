#!/usr/bin/env python3

import sys
sys.path.append('./package/')

from package import globalsVar as gl
from package import imagedisplay as imagedisplayer
from package import imageanalysis as imageanalysiser
from package import imagemanagement as imagemanager
from package import imagepreprocessing as imagepreprocesser


from package import autoencodertfmain as autoencodertfmainer
from package import conautoencodermain as conautoencodermainer

from package import cnntfmain as cnntfmainer

def test():
    gl._test()
    imagedisplayer._test()
    imageanalysiser._test1()
    imagemanager._test2()
    imagepreprocesser._test1()
    
def image_processing_main():
    imagemanager.main()
    imageanalysiser.main()
    pass

def image_processing_main_ela():
    imagemanager.main_ela()
    imageanalysiser.main_ela()
    pass

def main_preprocess_autoencoder():
    imagepreprocesser.main_autoencoder()

def main_preprocess_classification():
    imagepreprocesser.main_classification()

if __name__ == '__main__':
    
    ## image type
    image_processing_main()
    ## image_processing_main_ela()
    
    
    ## pre processing mode, deep learning mode
    
    main_preprocess_autoencoder()
    autoencodertfmainer.main()
    
    main_preprocess_autoencoder()
    conautoencodermainer.main(1)
    
    main_preprocess_autoencoder()
    conautoencodermainer.main(2)
    
    main_preprocess_autoencoder()
    conautoencodermainer.main(3)
    
    
    main_preprocess_classification()
    cnntfmainer.main()
    
    
    ## ELA
    image_processing_main_ela()
    
    main_preprocess_autoencoder()
    autoencodertfmainer.main_ela()
    
    main_preprocess_autoencoder()
    conautoencodermainer.main_ela(1)
    
    main_preprocess_autoencoder()
    conautoencodermainer.main_ela(2)
    
    main_preprocess_autoencoder()
    conautoencodermainer.main_ela(3)
    
    
    main_preprocess_classification()
    cnntfmainer.main_ela()
    
    pass