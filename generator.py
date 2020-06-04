from generator_objects import train_datagen, val_datagen

train_image_generator = train_datagen.flow_from_directory (
    './data/train_imgs',
    target_size=(256, 256),
    color_mode='rgb',
    class_mode=None,
    batch_size=1,
    interpolation='nearest',
)

train_mask_generator = train_datagen.flow_from_directory (
    './data/train_masks',
    target_size=(256, 256),
    color_mode="grayscale",
    class_mode=None,
    batch_size=1,
    interpolation='nearest'
)

val_image_generator = val_datagen.flow_from_directory (
    './data/val_imgs',
    target_size=(256, 256),
    color_mode='rgb',
    class_mode=None,
    batch_size=1,
    interpolation='nearest'
)

val_mask_generator = val_datagen.flow_from_directory (
    './data/val_masks',
    target_size=(256, 256),
    color_mode="grayscale",
    class_mode=None,
    batch_size=1,
    interpolation='nearest'
)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)