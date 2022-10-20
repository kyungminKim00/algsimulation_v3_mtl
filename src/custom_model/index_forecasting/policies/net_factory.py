import header.index_forecasting.RUNHEADER as RUNHEADER


def net_factory():
    default_cnn = None

    if RUNHEADER.default_net == 'nature_cnn_A':
        from custom_model.index_forecasting.nets.nature_cnn_A import nature_cnn
        default_cnn = nature_cnn
    elif RUNHEADER.default_net == 'nature_cnn_B':
        from custom_model.index_forecasting.nets.nature_cnn_B import nature_cnn
        default_cnn = nature_cnn
    elif RUNHEADER.default_net == 'nature_cnn_D':  # batch_normalization: on, regularization: off
        from custom_model.index_forecasting.nets.nature_cnn_D import nature_cnn
        default_cnn = nature_cnn
    elif RUNHEADER.default_net == 'nature_cnn_E':  # batch_normalization: on, regularization: off
        from custom_model.index_forecasting.nets.nature_cnn_E import nature_cnn
        default_cnn = nature_cnn
    elif RUNHEADER.default_net == 'inception_resnet_v2_A':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_A import inception_resnet_v2
        default_cnn = inception_resnet_v2
    elif RUNHEADER.default_net == 'inception_resnet_v2_B':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_B import inception_resnet_v2
        default_cnn = inception_resnet_v2
    elif RUNHEADER.default_net == 'inception_resnet_v2_C':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_C import inception_resnet_v2
        default_cnn = inception_resnet_v2

    # Dummy batch_normalization: off, regularization: off
    elif RUNHEADER.default_net == 'inception_resnet_v2_Dummy':
        from custom_model.index_forecasting.nets.inception_resnet_v2_Dummy import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off
    elif RUNHEADER.default_net == 'inception_resnet_v2_D':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_D import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: off, regularization: off (Not implemented yet)
    elif RUNHEADER.default_net == 'inception_resnet_v2_E':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_E import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: on
    elif RUNHEADER.default_net == 'inception_resnet_v2_F':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_F import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v1 (Stem Sparse 512)
    elif RUNHEADER.default_net == 'inception_resnet_v2_G':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: L1 -D v1 (Stem Sparse 512)
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_L1':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_L1 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: L1 -D v1 (Stem Sparse 512)
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_L2':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_L2 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: L1 -D v1 (Stem Sparse 512)
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_L2_2':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_L2_2 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v1 (Stem Sparse 512) - input feature experimental
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_T1':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_T1 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v1 (Stem Sparse 512) - input feature experimental
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_T2':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_T2 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v1 (Stem Sparse 512) - input feature experimental
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_T3':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_T3 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v1 (Stem Sparse 512) - input feature experimental
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_T4':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_T4 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v1 (Stem Sparse 512, Reduced Res) - input feature experimental
    elif RUNHEADER.default_net == 'inception_resnet_v2_G_T5':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_G_T5 import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v1 (Stem Sparse 1024)
    elif RUNHEADER.default_net == 'inception_resnet_v2_H':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_H import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v2 (Stem Sparse 1024, late concat)
    elif RUNHEADER.default_net == 'inception_resnet_v2_I':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_I import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v2 (Stem Sparse 512, late concat)
    elif RUNHEADER.default_net == 'inception_resnet_v2_I_B':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_I_B import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v3 (inceptionv4 + resnet keep more like original structure)
    # (Not implemented yet)
    elif RUNHEADER.default_net == 'inception_resnet_v2_J':
        from custom_model.index_forecasting.nets.inception_resnet_v2_IF_J import inception_resnet_v2
        default_cnn = inception_resnet_v2
    # batch_normalization: on, regularization: off -D v3 (inceptionv4 + resnet keep more like original structure)
    # (Not implemented yet)
    elif RUNHEADER.default_net == 'shake_regulization_v1':
        from custom_model.index_forecasting.nets.shake_regulization_v1 import shakenet
        default_cnn = shakenet
    elif RUNHEADER.default_net == 'shake_regulization_v2':
        from custom_model.index_forecasting.nets.shake_regulization_v2 import shakenet
        default_cnn = shakenet
    elif RUNHEADER.default_net == 'shake_regulization_v3':
        from custom_model.index_forecasting.nets.shake_regulization_v3 import shakenet
        default_cnn = shakenet
    elif RUNHEADER.default_net == 'shake_regulization_v4':
        from custom_model.index_forecasting.nets.shake_regulization_v4 import shakenet
        default_cnn = shakenet
    elif RUNHEADER.default_net == 'shake_regulization_v5':  # revise v3 - self attention part
        from custom_model.index_forecasting.nets.shake_regulization_v5 import shakenet
        default_cnn = shakenet
    else:
        ValueError('None Defined feature extractor')

    return default_cnn
