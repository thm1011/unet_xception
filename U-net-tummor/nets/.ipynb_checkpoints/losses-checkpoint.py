import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true[..., :-1])
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    # By setting the value of α > 𝜷, you can penalise false negatives more
    y_true_pos = K.flatten(y_true[..., :-1])
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def dice_loss_with_CE_weight(weight_map, beta=1, smooth = 1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1) * weight_map)

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE



def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)




def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


def tversky_loss_CEweight(weight_map, alpha=0.3):
    def _tversky_loss_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1) * weight_map)
        tv_loss = 1 - tversky(y_true, y_pred, alpha=alpha)
        return tv_loss + CE_loss
    return _tversky_loss_CE


def tversky_loss_CE(alpha=0.3):
    # By setting the value of alpha > 1-alpha, you can penalise false negatives more
    def _tversky_loss_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        tv_loss = 1 - tversky(y_true, y_pred, alpha=alpha)
        return tv_loss + CE_loss
    return _tversky_loss_CE

