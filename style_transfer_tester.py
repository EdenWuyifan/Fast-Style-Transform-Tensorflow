import tensorflow as tf
import transform

class StyleTransferTester:

    def __init__(self, session, content_image, model_path, batch_size):
        # session
        self.sess = session

        # input images
        self.x0 = content_image

        # input model
        self.model_path = model_path
        
        # batch size
        self.batch_size = batch_size

        # image transform network
        self.transform = transform.Transform()

        # build graph for style transfer
        self._build_graph()
        

    def _build_graph(self):

        # graph input
        batch_shape = (self.batch_size,512,512,3)
        self.x = tf.placeholder(tf.float32, shape=batch_shape, name='input')
        #self.xi = tf.expand_dims(self.x, 0) # add one dim for batch
        
        
        
        # result image from transform-net
        self.y_hat = self.transform.net(self.x/255.0)
        #self.y_hat = tf.squeeze(self.y_hat) # remove one dim for batch
        self.y_hat = tf.clip_by_value(self.y_hat, 0., 255.)

    def test(self):

        # initialize parameters
        self.sess.run(tf.global_variables_initializer())

        # load pre-trained model
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

        # get transformed image
        output = self.sess.run(self.y_hat, feed_dict={self.x: self.x0})

        return output





