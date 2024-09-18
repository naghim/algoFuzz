class AlgofuzzException(Exception):
    pass

class NotTrainedException(AlgofuzzException):

    def __init__(self):
        AlgofuzzException.__init__(self, 'You need to train the model first.')
