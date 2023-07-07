from fastai.callback.core import Callback


# Can this be a transform?
class DihedralCallback(Callback):
    def before_batch(self):
        """
        x: (batch_size, c, d, h, w)
        """
        xb = self.xb[0]
        yb = self.yb[0]

        k = random.randint(0,7)

        if k in [1,3,4,7]: 
            xb = xb.flip(-1)
            yb = yb.flip(-1)
        
        if k in [2,4,5,7]:
            xb = xb.flip(-2)
            yb = yb.flip(-2)

        if k in [3,5,6,7]: 
            xb = xb.transpose(-1,-2)
            yb = yb.transpose(-1,-2)

        self.learn.xb = (xb,)
        self.learn.yb = (yb,)
