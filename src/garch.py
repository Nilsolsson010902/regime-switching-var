from dataclasses import dataclass
import pandas as pd
import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult


@dataclass
class GarchOutput:
    result: ARCHModelResult
    conditional_volatility: pd.Series


def fit_garch(returns: pd.Series,
    distribution: str = "t"
) -> GarchOutput:
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