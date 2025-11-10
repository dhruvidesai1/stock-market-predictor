import numpy as np
import pandas as pd
import yfinance as yf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

DEFAULT_WINDOW = 20

def fetch_close_series(ticker: str, start_date: str, end_date: str):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty or 'Close' not in df.columns:
        raise ValueError(f"No data for {ticker} in range {start_date} - {end_date}")
    return df[['Close']].dropna()

def make_sequences(series, window=DEFAULT_WINDOW):
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_mlp(window):
    model = Sequential()
    model.add(InputLayer(input_shape=(window,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    return model

def build_lstm(window):
    model = Sequential()
    model.add(InputLayer(input_shape=(window,1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    return model

def flatten_weights(weights_list):
    flat = np.concatenate([w.flatten() for w in weights_list]).astype(np.float32)
    return flat

def unflatten_to_shapes(flat_vector, shapes):
    arrays = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        part = flat_vector[idx:idx+size]
        arrays.append(part.reshape(shape))
        idx += size
    return arrays

def get_model_flat_weights(model):
    weights = model.get_weights()
    shapes = [w.shape for w in weights]
    flat = flatten_weights(weights)
    return flat, shapes

def set_model_weights_from_flat(model, flat_vector, shapes):
    arrays = unflatten_to_shapes(flat_vector, shapes)
    model.set_weights(arrays)

def run_ga_for_model(X_train, y_train, model_factory,
                     population=12, ngen=6, cxpb=0.5, mutpb=0.2, mut_sigma=0.5, tournament=3,
                     random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    sample_model = model_factory()
    base_weights = sample_model.get_weights()
    shapes = [w.shape for w in base_weights]
    flat_base = flatten_weights(base_weights)
    num_weights = len(flat_base)

    # Ensure DEAP creator does not already exist
    try:
        creator.FitnessMin
    except Exception:
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    try:
        creator.Individual
    except Exception:
        creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('attr_float', lambda: random.uniform(-1, 1))
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_weights)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=mut_sigma, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=tournament)

    def evaluate(individual):
        m = model_factory()
        set_model_weights_from_flat(m, np.array(individual, dtype=np.float32), shapes)
        preds = m.predict(X_train, verbose=0)
        mse = np.mean((y_train - preds.flatten())**2)
        K.clear_session()
        return (mse,)

    toolbox.register('evaluate', evaluate)

    pop = toolbox.population(n=population)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring + pop, k=population)
        best = tools.selBest(pop, k=1)[0]
        print(f"[GA] Gen {gen+1}/{ngen} best MSE: {best.fitness.values[0]:.6f}")

    best_ind = tools.selBest(pop, k=1)[0]
    best_flat = np.array(best_ind, dtype=np.float32)
    return best_flat, shapes

def predict_with_ga_ann(ticker: str, start_date: str, end_date: str,
                        model_type='mlp', window=DEFAULT_WINDOW,
                        ga_population=12, ga_ngen=6,
                        fine_tune=False, fine_tune_epochs=5, random_seed=None):
    df = fetch_close_series(ticker, start_date, end_date)
    X, y, scaler = make_sequences(df['Close'], window=window)
    if len(X) < 10:
        raise ValueError('Not enough data after windowing; increase date range or reduce window.')

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if model_type == 'lstm':
        X_train_in = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_in = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model_factory = lambda: build_lstm(window)
    else:
        X_train_in = X_train
        X_test_in = X_test
        model_factory = lambda: build_mlp(window)

    best_flat, shapes = run_ga_for_model(X_train_in, y_train, model_factory,
                                        population=ga_population, ngen=ga_ngen, random_seed=random_seed)

    final_model = model_factory()
    set_model_weights_from_flat(final_model, best_flat, shapes)

    if fine_tune:
        final_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        final_model.fit(X_train_in, y_train, epochs=fine_tune_epochs, batch_size=16, verbose=1)

    preds_test = final_model.predict(X_test_in, verbose=0).flatten()
    preds_test_rescaled = scaler.inverse_transform(preds_test.reshape(-1,1)).flatten()
    actual_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    all_dates = df.index[window:]
    test_dates = all_dates[split_idx:]

    mse = float(np.mean((actual_test_rescaled - preds_test_rescaled)**2))
    rmse = float(np.sqrt(mse))

    K.clear_session()

    return {
        'ticker': ticker,
        'dates': [d.strftime('%Y-%m-%d') for d in test_dates],
        'actual': actual_test_rescaled.tolist(),
        'predicted': preds_test_rescaled.tolist(),
        'mse': mse,
        'rmse': rmse
    }
