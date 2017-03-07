import pickle
import os

class dataHandler():
    def __init__(self):
        self.data = {"frame"        :{},
                     "qVal"         :{},
                     "action"       :{},
                     "reward"       :{},
                     "isTerminal"   :{},
                     "internal1"    :{},
                     "internal2"    :{}}
        self.qty    = 0

        self.dir = os.path.join(os.path.dirname(__file__), './dataset/')


    def emptyHandler(self):
        self.data.clear()
        self.data = {"frame"        :{},
                     "qVal"         :{},
                     "action"       :{},
                     "reward"       :{},
                     "isTerminal"   :{},
                     "internal1"    :{},
                     "internal2"    :{}}
        self.qty = 0

    def addData(self,
                frame,
                expectedQ,
                action,
                reward,
                isTerminal,
                internal1,
                internal2):

        self.data["frame"][self.qty]      = frame
        self.data["qVal"][self.qty]       = expectedQ
        self.data["action"][self.qty]     = action
        self.data["reward"][self.qty]     = reward
        self.data["isTerminal"][self.qty] = isTerminal
        self.data["internal1"][self.qty]  = internal1
        self.data["internal2"][self.qty]  = internal2
        self.qty += 1

        if self.qty > 100:
            self.pickleSave()


    def pickleSave(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        with open(self.dir + 'dataset.pickle', 'a+b') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.emptyHandler()

    def pickleLoadNext(self):
        try:
            self.data = pickle.load(self.pickleLoad)
            self.qty    = len(self.data["isTerminal"])
            return True
        except EOFError:
            print("EOF")
            self.pickleLoad.close()
            return False
        except AttributeError:
            #pickle closed, open and retry
            self.pickleLoad = open(self.dir + 'dataset.pickle', 'rb')
            self.pickleLoadNext()
