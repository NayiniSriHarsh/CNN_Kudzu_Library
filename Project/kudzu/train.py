import numpy as np

class Learner:
    
    def __init__(self, loss, model, opt, epochs):
        self.loss = loss
        self.model = model
        self.opt = opt
        self.epochs = epochs
        self.cbs = []
        
    def set_callbacks(self, cblist):
        for cb in cblist:
            self.cbs.append(cb)
            
    def __call__(self, cbname, *args):
        status = True
        for cb in self.cbs:
            cbwanted = getattr(cb, cbname, None)
            status = status and cbwanted and cbwanted(*args)
        return status
    
    def train_loop(self, dl,X_train,y_train,test_input,y_test):
        self.dl = dl # dl added in here
        self.test_input = test_input
        bs = self.dl.bs
        datalen = len(self.dl.data)
        self.bpe = datalen//bs
        self.afrac = 0.
        if datalen % bs > 0:
            self.bpe  += 1
            self.afrac = (datalen % bs)/bs
        self('fit_start')
        for epoch in range(self.epochs):
            self('epoch_start', epoch)
            prediction = []
            for inputs, targets in dl:
                self("batch_start", dl.current_batch)
                
                # make predictions
                predicted = self.model(inputs)
                predicted_train = 1*(predicted>0.5)
                prediction.append(sum(1*(predicted_train == targets)))
                # actual loss value
                epochloss = self.loss(predicted, targets)
                self('after_loss', epochloss)

                # calculate gradient
                intermed = self.loss.backward(predicted, targets)
                self.model.backward(intermed)

                # make step
                self.opt.step(self.model)
                self('batch_end')
            train_predicted = self.model(X_train)
            predicted_train=1*(train_predicted>0.5)
            prob_train = np.sum((1*(predicted_train==y_train)))
            prob_train = prob_train/len(y_train)
            test_predicted = self.model(self.test_input)
            predicted_test=1*(test_predicted>0.5)
            prob_test = np.sum((1*(predicted_test==y_test)))
            prob_test = prob_test/len(y_test)
            print(f' epoch_accuracy:{prob_train} , epoch_test_accuracy:{prob_test}')
            self('epoch_end',predicted_train,prob_train,predicted_test,prob_test)
            
        self('fit_end')
        return epochloss