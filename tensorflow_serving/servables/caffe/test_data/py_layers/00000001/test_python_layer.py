import caffe

class TimesTenLayer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
      pass

    def reshape(self, bottom, top):
      top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
      top[0].data[...] = 10 * bottom[0].data

    def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = 10 * top[0].diff
