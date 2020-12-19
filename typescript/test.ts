import * as tf from '@tensorflow/tfjs';


const a = tf.tensor2d([[1, 2], [3, 4]]);
const b = tf.tensor3d([[[1]], [[3]]]);
const r = a.mul(b);

