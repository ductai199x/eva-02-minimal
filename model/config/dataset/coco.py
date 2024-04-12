


def get_train_config():
    return {
        "num_classes": 80,
        "n_dets_per_image": 100,
        "det_score_threshold": 0.05,
        "use_fed_loss": False,
        "get_fed_loss_cls_weights": None,
    }