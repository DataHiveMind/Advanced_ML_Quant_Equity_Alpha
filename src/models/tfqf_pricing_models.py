import tensorflow as tf
import tensorflow_quant_finance as tff


def calculate_implied_volatility(
    option_prices, strikes, expiries, spots, rates, is_call=True
):
    """
    Calculate implied volatility for European options using Black-Scholes model.
    :param option_prices: Tensor or array of observed option prices
    :param strikes: Tensor or array of strike prices
    :param expiries: Tensor or array of times to expiry (in years)
    :param spots: Tensor or array of spot prices
    :param rates: Tensor or array of risk-free rates
    :param is_call: Boolean, True for call options, False for puts
    :return: Tensor of implied volatilities
    """
    option_prices = tf.convert_to_tensor(option_prices, dtype=tf.float64)
    strikes = tf.convert_to_tensor(strikes, dtype=tf.float64)
    expiries = tf.convert_to_tensor(expiries, dtype=tf.float64)
    spots = tf.convert_to_tensor(spots, dtype=tf.float64)
    rates = tf.convert_to_tensor(rates, dtype=tf.float64)
    is_call = tf.convert_to_tensor(is_call, dtype=tf.bool)

    implied_vols = tff.black_scholes.implied_volatility.implied_volatility(
        option_prices=option_prices,
        forward=spots * tf.exp(rates * expiries),
        strike=strikes,
        expiry=expiries,
        discount_factor=tf.exp(-rates * expiries),
        is_call_options=is_call,
    )
    return implied_vols


def simulate_gbm_paths(spot, rate, volatility, expiry, num_paths, num_steps):
    """
    Simulate Geometric Brownian Motion (GBM) paths.
    :param spot: Initial spot price
    :param rate: Risk-free rate
    :param volatility: Volatility
    :param expiry: Time to expiry (in years)
    :param num_paths: Number of simulated paths
    :param num_steps: Number of time steps
    :return: Tensor of simulated paths (num_paths, num_steps + 1)
    """
    dt = expiry / num_steps
    times = tf.linspace(0.0, expiry, num_steps + 1)
    gbm = tff.models.GeometricBrownianMotion(
        mu=rate, sigma=volatility, dtype=tf.float64
    )
    paths = gbm.sample_paths(
        times=times,
        initial_state=tf.constant([spot], dtype=tf.float64),
        num_samples=num_paths,
        random_type=tff.math.random.RandomType.STATELESS,
        seed=(42, 42),
    )
    return tf.squeeze(paths, axis=-1)  # shape: (num_paths, num_steps + 1)


def price_european_option(strike, expiry, spot, volatility, rate, is_call=True):
    """
    Price a European option using Black-Scholes formula.
    :param strike: Strike price
    :param expiry: Time to expiry (in years)
    :param spot: Spot price
    :param volatility: Volatility
    :param rate: Risk-free rate
    :param is_call: Boolean, True for call, False for put
    :return: Option price
    """
    spot = tf.convert_to_tensor(spot, dtype=tf.float64)
    strike = tf.convert_to_tensor(strike, dtype=tf.float64)
    expiry = tf.convert_to_tensor(expiry, dtype=tf.float64)
    volatility = tf.convert_to_tensor(volatility, dtype=tf.float64)
    rate = tf.convert_to_tensor(rate, dtype=tf.float64)
    is_call = tf.convert_to_tensor(is_call, dtype=tf.bool)

    price = tff.black_scholes.option_price(
        volatilities=volatility,
        strikes=strike,
        expiries=expiry,
        spots=spot,
        discount_factors=tf.exp(-rate * expiry),
        is_call_options=is_call,
    )
    return
