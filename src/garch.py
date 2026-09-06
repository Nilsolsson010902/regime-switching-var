from dataclasses import dataclass
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


@dataclass
class GarchOutput:
    result: ARCHModelResult
    conditional_volatility: pd.Series


def fit_garch(returns: pd.Series, distribution: str = "t") -> GarchOutput:
    """
    Fit a GARCH(1,1) model to a return series.

    Parameters
    ----------
    returns:
        Daily returns expressed in percentage units.
    distribution:
        Innovation distribution. Common choices are
        "normal" and "t".

    Returns
    -------
    GarchOutput
        Fitted model result and conditional volatility series.
    """

    clean_returns = returns.dropna().astype(float)

    if clean_returns.empty:
        raise ValueError("The return series is empty after removing missing values.")

    model = arch_model(clean_returns, mean="Constant", vol="GARCH", p=1, q=1, dist=distribution, rescale=False)
    result = model.fit(disp="off")
    volatility = pd.Series(result.conditional_volatility, index=clean_returns.index, name="Conditional Volatility")

    return GarchOutput(result=result, conditional_volatility=volatility)


def forecast_variance_one_step(omega: float, alpha: float, beta: float, residual_t: float,variance_t: float) -> float:
    """
    Forecast the next period's variance using the GARCH(1,1) model.

    Parameters
    ----------
    omega:
        Constant term in the GARCH model.
    alpha:
        Coefficient for the lagged squared residual (ARCH term).
    beta:
        Coefficient for the lagged variance (GARCH term).
    residual_t:
        The residual (return - mean) at time t.
    variance_t:
        The variance at time t.

    Returns
    -------
    float
        Forecasted variance for time t+1.
    """
    return (omega+ alpha * residual_t**2 + beta * variance_t)