


def get_train_config():
    return {
        "num_classes": 1203,
        "n_dets_per_image": 300,
        "det_score_threshold": 0.02,
        "use_fed_loss": False, #TODO: Implement federated loss for LVIS
        "get_fed_loss_cls_weights": None,
    }