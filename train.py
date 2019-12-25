"""
Retrain the YOLO model for your own dataset.
"""
import tensorflow as tf
import datetime
import zipfile
from yolo3.model import darknet_yolo_body, YoloLoss, mobilenetv2_yolo_body, efficientnet_yolo_body
from yolo3.data import Dataset
from yolo3.enum import OPT, BACKBONE, DATASET_MODE
#from yolo3.map import MAPCallback
from yolo3.utils import get_classes, ModelFactory
import os
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

#tf.keras.backend.set_learning_phase(1)


def train(FLAGS):
    """Train yolov3 with different backbone
    """
    prune = FLAGS['prune']
    opt = FLAGS['opt']
    backbone = FLAGS['backbone']
    log_dir = FLAGS['log_directory'] or os.path.join(
        'logs',
        str(backbone).split('.')[1].lower() + str(datetime.date.today()))
    if tf.io.gfile.exists(log_dir) is not True:
        tf.io.gfile.mkdir(log_dir)
    batch_size = FLAGS['batch_size']
    train_dataset_glob = FLAGS['train_dataset']
    val_dataset_glob = FLAGS['val_dataset']
    test_dataset_glob = FLAGS['test_dataset']
    freeze = FLAGS['freeze']
    freeze_step = FLAGS['epochs'][0]
    train_step = FLAGS['epochs'][1]

    if opt == OPT.DEBUG:
        tf.config.experimental_run_functions_eagerly(True)
        tf.debugging.set_log_device_placement(True)
        tf.get_logger().setLevel(tf.logging.DEBUG)
    elif opt == OPT.XLA:
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)

    class_names = get_classes(FLAGS['classes_path'])
    num_classes = len(class_names)
    input_shape = FLAGS['input_size']  # multiple of 32, hw
    model_path = FLAGS['model']
    if model_path and model_path.endswith(
            '.h5') is not True:
        model_path = tf.train.latest_checkpoint(model_path)
    lr = FLAGS['learning_rate']
    tpu_address=FLAGS['tpu_address']
    if tpu_address is not None:
        cluster_resolver=tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        tf.config.experimental_connect_to_host(cluster_resolver.master())
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy=tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy(devices=FLAGS['gpus'])

    batch_size = batch_size * strategy.num_replicas_in_sync

    train_dataset_builder = Dataset(train_dataset_glob, batch_size,
                                     num_classes, input_shape)
    train_dataset, train_num = train_dataset_builder.build()
    val_dataset_builder = Dataset(val_dataset_glob,
                                  batch_size,
                                  num_classes,
                                  input_shape,
                                  mode=DATASET_MODE.VALIDATE)
    val_dataset, val_num = val_dataset_builder.build()
    #map_callback = MAPCallback(test_dataset_glob, input_shape,
    #                           class_names)
    logging = tf.keras.callbacks.TensorBoard(write_graph=False,
                                             log_dir=log_dir,
                                             write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(
        log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    monitor='val_loss',
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    period=2)
    cos_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.keras.experimental.CosineDecay(lr[1], train_step)(
            epoch - freeze_step).numpy(),1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=(freeze_step + train_step) // 10,
        verbose=0)
    if tf.version.VERSION.startswith('1.'):
        loss = [
            lambda y_true, yolo_output: YoloLoss(
                y_true, yolo_output, 0, print_loss=True), lambda
            y_true, yolo_output: YoloLoss(
                y_true, yolo_output, 1, print_loss=True)
        ]
    #else:
    #    loss = [YoloLoss(idx, print_loss=False) for idx in range(len(anchors) // 3)]

    with strategy.scope():
        #factory = ModelFactory(tf.keras.layers.Input(shape=(*input_shape, 3)),
        #                       weights_path=model_path)
        factory = ModelFactory(tf.keras.layers.Input(shape=(*input_shape, 3)))
        if backbone == BACKBONE.MOBILENETV2:
            model = factory.build(mobilenetv2_yolo_body,
                                  60,
                                  num_classes,
                                  alpha=1.0)
        #elif backbone == BACKBONE.DARKNET53:
        #    model = factory.build(darknet_yolo_body, 185,
        #                          len(anchors) // 3, num_classes)
        #elif backbone == BACKBONE.EFFICIENTNET:
        #    FLAGS['model_name'] = 'efficientnet-b4'
        #    model = factory.build(efficientnet_yolo_body,
        #                          100, # todo
        #                          FLAGS['model_name'],
        #                          len(anchors) // 2,
        #                          batch_norm_momentum=0.9,
        #                          batch_norm_epsilon=1e-3,
        #                          num_classes=num_classes,
        #                          drop_connect_rate=0.2,
        #                          data_format="channels_first")

    if prune:
        from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity
        end_step = np.ceil(1.0 * train_num / batch_size).astype(
            np.int32) * train_step
        new_pruning_params = {
            'pruning_schedule':
            sparsity.PolynomialDecay(initial_sparsity=0.5,
                                     final_sparsity=0.9,
                                     begin_step=0,
                                     end_step=end_step,
                                     frequency=1000)
        }
        pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
        pruned_model.compile(optimizer=tf.keras.optimizers.Adam(lr[0],
                                                                epsilon=1e-8),
                             loss=loss)
        pruned_model.fit(train_dataset,
                         epochs=train_step,
                         initial_epoch=0,
                         steps_per_epoch=max(1, train_num // batch_size),
                         callbacks=[
                             checkpoint, cos_lr, logging, early_stopping #map_callback
                         ],
                         validation_data=val_dataset,
                         validation_steps=max(1, val_num // batch_size))
        model = sparsity.strip_pruning(pruned_model)
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_pruned.h5'))
        with zipfile.ZipFile(os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_pruned.h5.zip'),
                             'w',
                             compression=zipfile.ZIP_DEFLATED) as f:
            f.write(
                os.path.join(
                    log_dir,
                    str(backbone).split('.')[1].lower() +
                    '_trained_weights_pruned.h5'))
        return

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if freeze is True:
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[0],
                                                             epsilon=1e-8),
                          loss=loss)
        model.fit(train_dataset,
                  epochs=freeze_step,
                  initial_epoch=0,
                  steps_per_epoch=max(1, train_num // batch_size),
                  callbacks=[logging, checkpoint],
                  validation_data=val_dataset,
                  validation_steps=max(1, val_num // batch_size))
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_stage_1.h5'))
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    #else:
    if 1:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[1],
                                                             epsilon=1e-8),
                          loss=loss)  # recompile to apply the change
        print('Unfreeze all of the layers.')
        model.fit(train_dataset,
                  epochs=train_step + freeze_step,
                  initial_epoch=freeze_step,
                  steps_per_epoch=max(1, train_num // batch_size),
                  callbacks=[
                      checkpoint,cos_lr, logging, early_stopping #map_callback
                  ],
                  validation_data=val_dataset,
                  validation_steps=max(1, val_num // batch_size))
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_final.h5'))

    # Further training if needed.
