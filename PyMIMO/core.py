class Processor():

    def __init__(self):
        self.name = "generic processor"

    def forward(self,X):
        return X

    def __call__(self,X):
        return self.forward(X)






