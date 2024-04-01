#!/usr/bin/env python3

import configparser
import os

class App:
    __conf = None

    @staticmethod
    def config():
        if App.__conf is None:  # Read once.
           ini_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'config.ini'))
           App.__conf = configparser.ConfigParser()
           App.__conf.read(ini_path)
        return App.__conf
    @staticmethod
    def get_config(section, option, fallback):
        if App.__conf is None:  # Read once.
           ini_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'config.ini'))
           App.__conf = configparser.ConfigParser()
           App.__conf.read(ini_path)
        # return App.__conf
        return App.config().get(section=section, option=option, fallback=fallback)

def _test():
    assert App.config().get(section='IMAGE', option='rawDataPath', fallback=None), "Configure file error"
    print(App.config().get(section='IMAGE', option='rawDataPath', fallback=3306))

if __name__ == '__main__':

   _test()
    

