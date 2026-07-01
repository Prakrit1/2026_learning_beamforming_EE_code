
from pathlib import Path
import gzip
import json
import pickle

import keras
import numpy as np

from src.models.helpers.network_models import PolicyNetworkSoft


def _load_policy_network_from_weights(
        model_path: Path,
) -> keras.Model:
    """
    CHECKPOINT SPEEDUP companion: reconstructs a PolicyNetworkSoft from the
    config saved alongside the checkpoint, then load_weights() into it. Pairs
    with save_model_checkpoint() in EE_sac.py, which switched from a full
    model.save() export to the much faster save_weights() -- see that
    function's comment for why. Only checkpoints produced by EE_sac.py use
    this weights-only format; load_model() below detects which format a given
    checkpoint is in, so older full-SavedModel checkpoints still load fine.
    """

    with open(Path(model_path, 'config', 'config_sac_learner.json')) as file:
        config_learner_dict = json.load(file)
    network_args = config_learner_dict['algorithm_args']['network_args']

    network = PolicyNetworkSoft(
        num_actions=network_args['num_actions'],
        **network_args['policy_network_args'],
    )
    dummy_state = np.zeros((1, network_args['size_state']), dtype='float32')
    network.initialize_inputs(dummy_state)
    network.load_weights(Path(model_path, 'model', 'weights.weights.h5'))

    return network


def load_model(
        model_path: Path,
) -> (keras.Model, dict):

    # CHECKPOINT SPEEDUP: detect weights-only (new, fast) vs full-SavedModel
    # (old) checkpoint format -- see _load_policy_network_from_weights above.
    weights_path = Path(model_path, 'model', 'weights.weights.h5')
    if weights_path.exists():
        network = _load_policy_network_from_weights(model_path)
    else:
        network = keras.models.load_model(Path(model_path, 'model'))

    with gzip.open(Path(model_path, 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']

    return network, norm_factors


def load_models(
        models_path: Path,
) -> (list[keras.Model], dict):

    paths = sorted([
        Path(path, 'model')
        for path in models_path.iterdir()
        if path.is_dir() and 'agent' in path.name
    ])
    networks = [keras.models.load_model(model_path) for model_path in paths]

    with gzip.open(Path(models_path, 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']

    return networks, norm_factors