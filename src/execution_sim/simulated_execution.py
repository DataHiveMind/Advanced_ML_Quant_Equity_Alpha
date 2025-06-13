import numpy as np


def calculate_slippage(order_size, current_volume, price_deviation_model=None):
    """
    Estimate slippage based on order size and current market volume.
    :param order_size: int, number of shares/contracts in the order
    :param current_volume: int, current market volume
    :param price_deviation_model: function or None, models price impact as a function of order/volume
    :return: float, slippage as a fraction of price (e.g., 0.001 = 0.1%)
    """
    if price_deviation_model is not None:
        return price_deviation_model(order_size, current_volume)
    # Simple square-root impact model as default
    if current_volume == 0:
        return 0.01  # fallback for illiquid
    impact = 0.001 * np.sqrt(abs(order_size) / current_volume)
    return min(impact, 0.05)  # cap slippage at 5%


def execute_order(order_details, current_market_data, market_impact_model=None):
    """
    Simulate order execution considering order type, market impact, and liquidity.
    :param order_details: dict with keys 'symbol', 'side', 'size', 'order_type', 'limit_price' (optional)
    :param current_market_data: dict with keys 'price', 'volume', 'bid', 'ask'
    :param market_impact_model: function or None, models price impact
    :return: dict with keys 'filled_size', 'avg_fill_price', 'slippage'
    """
    symbol = order_details["symbol"]
    side = order_details["side"]  # 'buy' or 'sell'
    size = order_details["size"]
    order_type = order_details.get("order_type", "market")
    limit_price = order_details.get("limit_price", None)
    price = current_market_data["price"]
    volume = current_market_data["volume"]
    bid = current_market_data.get("bid", price)
    ask = current_market_data.get("ask", price)

    # Determine fill price and size
    if order_type == "market":
        slippage = calculate_slippage(size, volume, market_impact_model)
        fill_price = ask if side == "buy" else bid
        fill_price *= (1 + slippage) if side == "buy" else (1 - slippage)
        filled_size = size
    elif order_type == "limit":
        if (side == "buy" and limit_price >= ask) or (
            side == "sell" and limit_price <= bid
        ):
            slippage = calculate_slippage(size, volume, market_impact_model)
            fill_price = ask if side == "buy" else bid
            fill_price *= (1 + slippage) if side == "buy" else (1 - slippage)
            filled_size = size
        else:
            fill_price = None
            filled_size = 0
            slippage = 0.0
    else:
        raise ValueError(f"Unknown order type: {order_type}")

    return {
        "filled_size": filled_size,
        "avg_fill_price": fill_price,
        "slippage": slippage,
    }
