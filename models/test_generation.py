import tensorflow as tf
import os
import time
import numpy as np
from utils import *
from ops import *
import cv2
import matplotlib.pyplot as plt

def test_generation(model, sess):
    config = model.config
    bs = config.batch_size
    n_batches = 1
    n_split = 1
    filesize = bs*n_batches/n_split

    global_step = tf.Variable(0, trainable=False)
    is_training = False
    d_optim, g_optim = model.build_model(is_training, global_step=global_step)
    res = model.load(sess, config.load_cp_dir)
    print res

    print "[*] Test Start"
    start_time = time.time()

    samples, _ = generate_memgan(model, sess, n_iter=n_batches, batch_size=bs)
    print(samples.shape)

    total_time = time.time() - start_time
    print "[*] Finished : %f" % (total_time)

    image_dir = os.path.join('samples', config.load_cp_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    save_images(samples, [8, 8], os.path.join(image_dir, 'samples.png'))

    for i in range(n_split):
        with open(os.path.join(image_dir, 'samples_gen_%d.npy' % i), 'w') as f:
            np.save(f, samples[i*filesize:(i+1)*filesize])

    print "saved at %s" % image_dir
    sess.close()


def generalization_examples(model, sess):
    config = model.config
    bs = config.batch_size
    num_comparisons = 8  # Do this for x different generated images
    k_nn = 7  # the seven nearest neighbors in memory

    global_step = tf.Variable(0, trainable=False)
    is_training = False
    d_optim, g_optim = model.build_model(is_training, global_step=global_step)
    res = model.load(sess, config.load_cp_dir)
    print(res)

    # build key matrix of examples from cifar10
    keys, images = get_keys_for_dataset(model, n_batches=100)
    gen_img, z = generate_memgan(model, sess, n_iter=1, batch_size=bs)
    gen_key = model.q_f.eval(feed_dict={model.image: gen_img, model.z: z})
    gen_img = inverse_transform(gen_img)

    final_images = []
    for i in range(num_comparisons):
        similarity = np.dot(keys/np.linalg.norm(keys, axis=1)[:, np.newaxis],
                            gen_key[i]/np.linalg.norm(gen_key[i]))
        top_k = similarity.argsort()[-k_nn:][::-1]

        final_images.append(np.expand_dims(gen_img[i], axis=0))
        for j in range(k_nn):
            final_images.append(np.expand_dims(images[top_k[j]], axis=0))

    final_images_matrix = np.concatenate(final_images, axis=0)

    save_image_sample(final_images_matrix)
    sess.close()


def generate_memgan(model, sess, n_iter=100, batch_size=64):
    samples = []

    for i in range(n_iter):
        z = get_z(model, batch_size)
        sample = sess.run(model.gen_image, feed_dict={model.z: z})
        print('done sampling...')
        samples.append(sample)

    return np.concatenate(samples, axis=0), z


def get_keys_for_dataset(model, n_batches=100):
    keys = []
    images = []
    dataset = load_dataset(model)
    for idx in xrange(0, n_batches):
        z = get_z(model, model.batch_size)
        image, label = dataset.next_batch(model.batch_size)

        keys.append(model.q_r.eval(feed_dict={model.image: image,
                                        model.label: label,
                                        model.z: z}))
        images.append(image)

    return np.concatenate(keys, axis=0), np.concatenate(images, axis=0)


def load_dataset(model):
    if model.dataset_name == 'mnist':
        import mnist as ds
    elif model.dataset_name == 'fashion':
        import fashion as ds
    elif model.dataset_name == 'affmnist':
        import affmnist as ds
    elif model.dataset_name == 'cifar10':
        import cifar10 as ds
    elif model.dataset_name == 'celeba':
        import celeba as ds
    elif model.dataset_name == 'chair':
        import chair as ds
    return ds.read_data_sets(model.dataset_path, dtype=tf.uint8, reshape=False, validation_size=0).train


def get_z(model, batch_size=64):
    return np.random.uniform(-1., 1., size=(batch_size, model.z_dim))


def save_image_sample(images):
    f, axarr = plt.subplots(nrows=int(np.sqrt(len(images))), ncols=int(np.sqrt(len(images))))
    indx = 0
    for i in range(int(np.sqrt(len(images)))):
        for j in range(int(np.sqrt(len(images)))):
            if j != 0:
                image = images[indx].astype("uint8")
            else:
                image = images[indx]
            axarr[i, j].imshow(image)
            indx += 1

            # Turn off tick labels
            axarr[i, j].axis('off')

    f.tight_layout()
    plt.subplots_adjust(wspace=0.10, hspace=0.10)
    directory = 'samples/checkpoint/pretrained_model_ours'
    f.savefig(directory+'/generalization_images')
