# The key deliverable of Task C.4 is build_model(): a single function that
# accepts the architectural parameters of a recurrent network and returns a
# compiled Keras Sequential model — no manual layer-by-layer coding required.
#
# Supported recurrent cell types: LSTM, GRU, SimpleRNN.
# References
# ----------
# [1] Keras Sequential model API
#     https://keras.io/guides/sequential_model/
# [2] Keras LSTM layer
#     https://keras.io/api/layers/recurrent_layers/lstm/
# [3] Keras GRU layer
#     https://keras.io/api/layers/recurrent_layers/gru/
# [4] Keras SimpleRNN layer
#     https://keras.io/api/layers/recurrent_layers/simple_rnn/
# [5] Dropout as regularisation
#     https://jmlr.org/papers/v15/srivastava14a.html
# [6] Stacked / deep RNNs
#     https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Input


# ── Task C.4 core function ───────────────────────────────────────────────────

def build_model(
        sequence_length: int,
        layer_sizes: list[int],
        layer_type: str = 'LSTM',
        dropout_rate: float = 0.2,
        dense_units: int = 1,
        n_features: int = 1,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error'
    ):
    """
    Build and compile a stacked recurrent neural network.

    This function fulfils the Task C.4 requirement: *write a function that takes
    as input the number of layers, the size of each layer, and the layer name
    and returns a Deep Learning model.*

    Parameters
    ----------
    sequence_length : int
        Number of past time-steps fed to the model at each forward pass
        (= PREDICTION_DAYS in main.py).  Used to declare the Input shape.

    layer_sizes : list[int]
        Length of this list determines the number of recurrent layers.
        Each element sets the number of units (neurons) in that layer.
        Example: [128, 64, 32] → three recurrent layers of decreasing width.

    layer_type : str
        Name of the recurrent cell to use.  Accepted values (case-insensitive):
          'LSTM'      – Long Short-Term Memory [2]
          'GRU'       – Gated Recurrent Unit [3]
          'RNN'       – Simple (Elman) RNN [4]
        Any other string raises a ValueError.

    dropout_rate : float
        Fraction of units randomly zeroed during each training step.
        Dropout [5] is a widely-used regularisation technique that prevents
        co-adaptation of neurons and reduces overfitting.
        Set to 0.0 to disable dropout entirely.

    dense_units : int
        Width of the final Dense output layer.  For single-step price
        prediction this is 1; for multistep prediction it is k (Task C.5).

    n_features : int
        Number of input feature time-series per timestep (Task C.5).
        1 for univariate (Close only); len(OHLCV) for the multivariate problem.
        Sets the last dimension of the Input shape.

    optimizer : str
        Keras optimiser name, e.g. 'adam', 'rmsprop', 'sgd'.
        Adam is the default because it combines momentum and adaptive learning
        rates and converges reliably on most sequence tasks.

    loss : str
        Keras loss function name.  'mean_squared_error' (MSE) is standard for
        regression; 'mean_absolute_error' is more robust to outliers.

    Returns
    -------
    model : keras.Sequential
        A compiled model ready for .fit().

    How the stacking works
    ----------------------
    All recurrent layers except the last one must pass their *full output
    sequence* to the next layer, not just the final hidden state.  This is
    done by setting return_sequences=True on every layer except the last.
    Without it the shape collapses to (batch, units) after the first layer
    and subsequent recurrent layers have no sequence to iterate over. [6]

    Layer-by-layer Dropout is added after every recurrent layer (when
    dropout_rate > 0) to regularise the weights of that layer independently.
    """

    # ── Resolve the recurrent layer class ────────────────────────────────────
    layer_type_upper = layer_type.strip().upper()
    layer_registry = {
        'LSTM': LSTM,
        'GRU':  GRU,
        'RNN':  SimpleRNN,
    }
    if layer_type_upper not in layer_registry:
        raise ValueError(
            f"Unknown layer_type '{layer_type}'. "
            f"Choose from: {list(layer_registry.keys())}"
        )
    RecurrentLayer = layer_registry[layer_type_upper]

    # ── Build the Sequential graph ────────────────────────────────────────────
    model = Sequential()

    # Input layer: explicitly declares the shape (sequence_length, n_features).
    # n_features=1 is the univariate case; >1 feeds several feature series at each
    # timestep (the multivariate problem). Using keras.layers.Input avoids the
    # deprecation warning that arises when input_shape is passed directly to the
    # first recurrent layer in newer Keras.
    model.add(Input(shape=(sequence_length, n_features)))

    # ── Recurrent layers ──────────────────────────────────────────────────────
    n_layers = len(layer_sizes)
    for i, units in enumerate(layer_sizes):
        # All layers except the last must return their full output sequence so
        # the next recurrent layer receives a 3-D tensor (batch, timesteps, units)
        # instead of just the last hidden state (batch, units).
        is_last_recurrent = (i == n_layers - 1)
        return_sequences  = not is_last_recurrent

        model.add(RecurrentLayer(units=units, return_sequences=return_sequences))

        # Dropout after every recurrent layer (skip when rate == 0).
        if dropout_rate > 0.0:
            model.add(Dropout(rate=dropout_rate))

    # ── Output layer ──────────────────────────────────────────────────────────
    # A plain Dense layer with linear activation maps the final hidden state to
    # the predicted price value(s).  No activation = identity function, which is
    # correct for regression (we do not want to squash the output to [0,1]).
    model.add(Dense(units=dense_units))

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(optimizer=optimizer, loss=loss)

    # Print a human-readable summary so the user can verify the architecture.
    model.summary()
    return model


# ── Helper Functions ──────────────────────────────────────────────────────

def get_model_path(model_dir: str, company: str, tag: str = '') -> str:
    """Return a .keras file path, optionally distinguished by a tag string."""
    os.makedirs(model_dir, exist_ok=True)
    suffix = f'_{tag}' if tag else ''
    return os.path.join(model_dir, f'{company}{suffix}_model.keras')


def load_or_build(
        model_path: str,
        build_kwargs: dict,
        x_train,
        y_train,
        epochs: int = 25,
        batch_size: int = 32,
        force_retrain: bool = False
    ):
    """
    Return a trained model: load from disk if it exists, otherwise build,
    train, and save it.

    Parameters
    ----------
    model_path   : Full path to the .keras file.
    build_kwargs : Keyword arguments forwarded verbatim to build_model().
    x_train      : Training input array (samples, timesteps, 1).
    y_train      : Training target array (samples,).
    epochs       : Number of full passes through the training set.
    batch_size   : Number of samples per gradient update.
    force_retrain: Re-build and retrain even if a saved model exists.
    """
    if os.path.exists(model_path) and not force_retrain:
        print(f"[model_builder] Loading saved model from {model_path} …")
        return load_model(model_path)

    print(f"[model_builder] Building new {build_kwargs.get('layer_type','LSTM')} model …")
    model = build_model(**build_kwargs)

    print(f"[model_builder] Training for {epochs} epochs, batch_size={batch_size} …")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    model.save(model_path)
    print(f"[model_builder] Model saved to {model_path}")
    return model
