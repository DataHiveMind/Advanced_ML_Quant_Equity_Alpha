import QuantLib as ql
import numpy as np


def build_yield_curve(deposits, swaps, settlement_days=2, calendar=ql.TARGET()):
    """
    Build a QuantLib yield curve from deposit and swap rates.
    :param deposits: list of tuples (tenor_str, rate), e.g. [("1M", 0.01), ("3M", 0.012)]
    :param swaps: list of tuples (tenor_str, rate), e.g. [("2Y", 0.015), ("5Y", 0.02)]
    :param settlement_days: int, settlement lag in days
    :param calendar: QuantLib calendar
    :return: QuantLib YieldTermStructureHandle
    """
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    deposit_helpers = [
        ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate)),
            ql.Period(tenor),
            settlement_days,
            calendar,
            ql.ModifiedFollowing,
            False,
            ql.Actual360(),
        )
        for tenor, rate in deposits
    ]

    swap_helpers = [
        ql.SwapRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate)),
            ql.Period(tenor),
            calendar,
            ql.Annual,
            ql.Unadjusted,
            ql.Thirty360(),
            ql.Euribor6M(),
        )
        for tenor, rate in swaps
    ]

    rate_helpers = deposit_helpers + swap_helpers
    yield_curve = ql.PiecewiseLinearZero(today, rate_helpers, ql.Actual365Fixed())
    return ql.YieldTermStructureHandle(yield_curve)


def price_european_option_ql(option_data):
    """
    Price a European option using QuantLib Black-Scholes-Merton process.
    :param option_data: dict with keys 'spot', 'strike', 'expiry' (in years), 'vol', 'rate', 'is_call'
    :return: float, option price
    """
    spot = ql.SimpleQuote(option_data["spot"])
    rate = ql.SimpleQuote(option_data["rate"])
    vol = ql.SimpleQuote(option_data["vol"])
    today = ql.Date.todaysDate()
    expiry_date = today + int(option_data["expiry"] * 365)

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_data.get("is_call", True) else ql.Option.Put,
        option_data["strike"],
    )
    exercise = ql.EuropeanExercise(expiry_date)

    spot_handle = ql.QuoteHandle(spot)
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, ql.QuoteHandle(rate), ql.Actual365Fixed())
    )
    flat_vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            today, ql.TARGET(), ql.QuoteHandle(vol), ql.Actual365Fixed()
        )
    )

    bsm_process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol)
    option = ql.VanillaOption(payoff, exercise)
    price = option.NPV(bsm_process)
    return price


def calculate_bond_duration(bond_details, yield_curve):
    """
    Calculate the Macaulay duration of a fixed-coupon bond.
    :param bond_details: dict with keys 'face', 'coupon', 'maturity' (years), 'frequency' (e.g., ql.Annual)
    :param yield_curve: QuantLib YieldTermStructureHandle
    :return: float, Macaulay duration
    """
    today = ql.Date.todaysDate()
    maturity_date = today + int(bond_details["maturity"] * 365)
    schedule = ql.Schedule(
        today,
        maturity_date,
        ql.Period(bond_details.get("frequency", ql.Annual)),
        ql.TARGET(),
        ql.Following,
        ql.Following,
        ql.DateGeneration.Backward,
        False,
    )
    bond = ql.FixedRateBond(
        0,  # settlement days
        bond_details["face"],
        schedule,
        [bond_details["coupon"]],
        ql.ActualActual(),
    )
    bond.setPricingEngine(ql.DiscountingBondEngine(yield_curve))
    macaulay_duration = ql.BondFunctions.duration(
        bond, yield_curve, ql.Duration.Macaulay
    )
    return
