CONFIGURATION = {
    "N_MAPS" : 4,
    "N_PATCHES": (256 // 16) * (256 // 16),
    "HIDDEN_SIZE": 64,

    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 20,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
    "NUM_CLASSES": 7,
    "PATCH_SIZE": 16,
    "PROJ_DIM": 768,
    "CLASS_NAMES": ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC'],
}

