from utils import constants
import importlib
import time
from datetime import datetime
import os
import numpy as np
from facenet import face_net, lfw
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import itertools


def main(data_dir, log_dir, model_dir, batch_size_train, epoch_size_train):
    """
    main process for tripless lot train
    :return:
    """
    batch_size_train = int(batch_size_train)
    epoch_size_train = int(epoch_size_train)
    network = importlib.import_module(constants.MODEL_DEF_DEFAULT)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    logdir = log_dir
    print(logdir)

    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    #model_dir = os.path.join(os.path.expanduser(constants.MODELS_BASE_DIR_DEFAULT), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    np.random.seed(seed=constants.SEED_DEFAULT)

    # Load traing set
    print('Data dir : {0}'.format(data_dir))
    train_set = face_net.load_dataset(data_dir)

    print('Model dir : {0}'.format(model_dir))
    print('Log dir : {0}'.format(logdir))

    # TODO load pre-trained model

    # Load LFW
    if constants.LFW_DIR_DEFAULT:
        print('LFW dir : {0}'.format(constants.LFW_DIR_DEFAULT))
        # Read the file which contain the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(constants.LFW_PAIRS_DEFAULT))
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(constants.LFW_DIR_DEFAULT), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(constants.SEED_DEFAULT)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='labels')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None,
                                              name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_thread = constants.LFW_NROF_PREPROCESS_THREAD_DEFAULT
        images_and_labels = []
        for i in range(nrof_preprocess_thread):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_content = tf.read_file(filename)
                image = tf.to_float(tf.image.decode_image(file_content, channels=3))

                if constants.RANDOM_CROP_DEFAULT:
                    image = tf.random_crop(image, [constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, constants.IMAGE_SIZE, constants.IMAGE_SIZE)

                if constants.RANDOM_FLIP_DEFAULT:
                    image = tf.image.random_flip_left_right(image)

                # Pylink : disable=no-member
                image.set_shape((constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
                images.append(image)
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size_placeholder,
            shapes=[(constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3), ()],
            enqueue_many=True,
            capacity=4 * nrof_preprocess_thread * batch_size_train,
            allow_smaller_final_batch=True)

        # TODO something strange
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Buld the inference graph
        prelogits, _ = network.inference(image_batch, constants.KEEP_PROBABILITY_DEFAULT,
                                         phase_train=phase_train_placeholder,
                                         bottleneck_layer_size=constants.EMBEDDING_SIZE_DEFAULT,
                                         weight_decay=constants.WEIGHT_DECAY_DEFAULT)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, constants.EMBEDDING_SIZE_DEFAULT]), 3, 1)
        triplet_loss = face_net.triplet_lot(anchor, positive, negative, constants.ALPHA_DEFAULT)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   constants.LEARNING_RATE_DECAY_EPOCHS_DEFAULT * epoch_size_train,
                                                   constants.LEARNING_RATE_DECAY_EPOCHS_FACTOR, staircase=True)
        tf.summary.scalar('learing_rate', learning_rate)

        # Calculate the total losses
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_loss, name='total_loss')

        # Build a Graph that train the model with one batch of example and update the model parameter
        train_op = face_net.train(total_loss, global_step, constants.OPTIMIZER_DEFAULT, learning_rate,
                                  constants.MOVING_AVERAGE_DECAY_DEFAULT, tf.global_variables(), log_dir)

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of summaries
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=constants.GPU_MEMORY_FRACTION_DEFAULT)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if constants.PRETRAINED_MODEL_DEFAULT:
                print('Restoring pretrained model: {0}'.format(constants.PRETRAINED_MODEL_DEFAULT))
                saver.restore(sess, os.path.expanduser(constants.PRETRAINED_MODEL_DEFAULT))

            # Training and validation loop
            epoch = 0
            while epoch < constants.MAX_NROF_EPOCHS_DEFAULT:
                step = sess.run(global_step, feed_dict=None)
                epoch = step / epoch_size_train
                print('epoch for train {0}'.format(epoch))

                # Training for each one epoch
                train(sess, batch_size_train, epoch_size_train, train_set, epoch, image_paths_placeholder, labels_placeholder,
                      labels_batch, batch_size_placeholder, learning_rate_placeholder,
                      phase_train_placeholder, enqueue_op, input_queue, global_step,
                      embeddings, total_loss, train_op, summary_op, summary_writer,
                      constants.LEARNING_RATE_SCHEDULE_FILE_DEFAULT, constants.EMBEDDING_SIZE_DEFAULT,
                      anchor, positive, negative, triplet_loss)

                # Save variables and the metagraph if it does not exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LW
                if constants.LFW_DIR_DEFAULT:
                    evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                             actual_issame, batch_size_train, constants.LFW_NROF_FOLDS_DEFAULT, logdir, step,
                             summary_writer, constants.EMBEDDING_SIZE_DEFAULT)

    print('End main process.')
    return model_dir


def train(sess, batch_size_train, epoch_size_train, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch, batch_size_placeholder,
          learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, embeddings,
          total_loss, train_op, summary_op, summary_writer, learning_rate_schedule_file_default, embedding_size_default,
          anchor, positive, negative, triplet_loss):
    """

    Train model

    :return:
    """

    batch_number = 0

    if constants.LEARNING_RATE_DEFAULT > 0.0:
        lr = constants.LEARNING_RATE_DEFAULT
    else:
        lr = face_net.get_learning_rate_from_file(learning_rate_schedule_file_default, epoch)

    while batch_number < epoch_size_train:
        # Same people randomly from the dataset
        image_paths, num_per_class = sample_people(train_set, constants.PEOPLE_PER_BATCH_DEFAULT, constants.IMAGES_PER_PERSON_DEFAULT)

        print('Running forward pass on sample images: ', end='')

        start_time = time.time()
        nrof_examples = constants.PEOPLE_PER_BATCH_DEFAULT * constants.IMAGES_PER_PERSON_DEFAULT
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))

        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

        emb_array = np.zeros((nrof_examples, embedding_size_default))
        nrof_batches = int(np.ceil(nrof_examples / batch_size_train))

        for i in range(nrof_batches):
            batch_size = min(nrof_examples - (i * batch_size_train), batch_size_train)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                       learning_rate_placeholder: lr,
                                                                       phase_train_placeholder: True})
            emb_array[lab, :] = emb

        print('%.3f' % (time.time() - start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                    image_paths, constants.PEOPLE_PER_BATCH_DEFAULT,
                                                                    constants.ALPHA_DEFAULT)

        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil((nrof_triplets * 3) / batch_size_train))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))

        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})

        nrof_examples = len(triplet_paths)

        emb_array = np.zeros((nrof_examples, embedding_size_default))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()

        train_time = 0
        i = 0
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - i * batch_size_train, batch_size_train)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr,
                         phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([total_loss, train_op, global_step, embeddings, labels_batch],
                                              feed_dict=feed_dict)
            emb_array[lab, :] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number + 1, epoch_size_train, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)

        # Add validation loss and accuracy to summary
        # pylink: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)

    return step


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-{0}.ckpt'.format(model_name))
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in {0} seconds'.format(save_time_variables))
    metagrapph_filename = os.path.join(model_dir, 'model-{0}.meta'.format(model_name))
    save_time_metagraph = 0

    if not os.path.exists(metagrapph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagrapph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in {0} seconds'.format(save_time_metagraph))

    summary = tf.Summary()
    # pylink: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagrahp', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame,
             batch_size, nrof_folds, logdir, step, summary_writer, embedding_size_default):
    start_time = time.time()
    print('Running forward pass on LFE images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size_default))
    nrof_batches = int(np.ceil(nrof_images, ))
    label_check_array = np.zeros((nrof_images))

    for i in range(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                   learning_rate_placeholder: 0.0,
                                                                   phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1

    print('{0}'.format(time.time() - start_time))

    assert (np.all(label_check_array == 1))

    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    lfw_time = time.time() - start_time

    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylink: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(logdir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


def sample_people(train_set, people_per_batch, images_per_person):
    """
    Sample people
    """
    nrof_images = people_per_batch * images_per_person
    #nrof_images = facenet.calculate_images(train_set)

    # Sample classes from the dataset
    nrof_classes = len(train_set)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []

    # TODO Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(train_set[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))

        idx = image_indices[0: nrof_images_from_class]
        image_paths_for_class = [train_set[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def select_triplets(embedings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """
    Select triplet
    """

    trip_index = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1, nrof_images):
            a_index = emb_start_idx + j - 1
            neg_dist_sqr = np.sum(np.square(embedings[a_index] - embedings), 1)
            for pair in range(j, nrof_images):
                p_index = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embedings[a_index] - embedings[p_index]))
                neg_dist_sqr[emb_start_idx: emb_start_idx + nrof_images] = np.NaN

                # TODO difference between Facenet and VGG
                # FaceNet selection
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]

                # VGG Face selecction
                all_neg = np.where(neg_dist_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_index = np.random.randint(nrof_random_negs)
                    n_index = all_neg[rnd_index]
                    triplets.append((image_paths[a_index], image_paths[p_index], image_paths[n_index]))
                    trip_index += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


if __name__ == '__main__':
    main(constants.DATA_DIR_DEFAULT, constants.LOGS_BASE_DIR_DEFAULT, constants.MODELS_BASE_DIR_DEFAULT, constants.BATCH_SIZE, constants.EPOCH_SIZE_DEFAULT)
