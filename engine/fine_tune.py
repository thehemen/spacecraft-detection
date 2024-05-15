class FineTuneScheduler:
    def __init__(self, layer_num, max_epoch):
        self.layer_num = layer_num
        self.max_epoch = max_epoch

        self.__frozen_layer_index = -1
        self.__indexes = []

    def init(self, model):
        layer_per_epoch = self.max_epoch / self.layer_num

        self.__frozen_layer_index = self.layer_num - 1
        self.__indexes = [int(round(x * layer_per_epoch)) for x in range(self.layer_num)]

        for i in range(self.layer_num):
            model.detector.features[i].requires_grad = False

    def step(self, index, model):
        if index not in self.__indexes:
            return False

        while index in self.__indexes:
            model.detector.features[self.__frozen_layer_index].requires_grad = False
            self.__indexes.remove(index)
            self.__frozen_layer_index -= 1

        return True
