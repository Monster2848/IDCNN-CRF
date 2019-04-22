# -!- coding: utf-8 -!-
from model import Crf_model

class Run(Crf_model):

    def __init__(self):
        Crf_model.__init__(self)
        if self.pattern == 'train':
            self.train()
        else:
            self.demo()


if __name__ == '__main__':
    Run()
