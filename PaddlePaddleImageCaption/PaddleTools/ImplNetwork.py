
class ImplNetwork():

    def get_input(self):
        raise NotImplementedError()

    def get_output(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def build(self):
        raise NotImplementedError()
