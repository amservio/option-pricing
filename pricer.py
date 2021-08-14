# all formulas from haug 2007
import math
from numpy import divide
from scipy.stats import norm

def BSMprice(type, S, X, T, r, v, d = 0, f = 0):
    """ Calculates the price of an european vanilla option using the Generalized
        Black-Scholes-Merton formula

    Parameters:
        type (string): 'c' for a Call Option or 'p' for a Put Option
        S (float): Current(spot) price for the underlying Stock
        X (float): The option's strike value
        T (int): Weekdays until expiration date
        r (float): Risk-Free interest rate for expiration date
        v (float): Implied volatility
        d (float): Annualized dividend yield for underlying stock
        f (float): Foreign currency risk-free interest rate (for corrency options)
    
    Returns:
        px (float): calculated option price for the closed-form Black-Scholes model
    """

    d1 = (math.log(S/X) + (r - f - d + v**2 / 2.0) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    
    if type == 'c':
        px = S * math.exp((-d -f) * T) * norm.cdf(d1) - \
            X * math.exp(-r * T) * norm.cdf(d2)
    else:
        px = X * math.exp(-r * T) * norm.cdf(-d2) - \
            S * math.exp((-d -f) * T) * norm.cdf(-d1)
    return(px)

def BSMdelta(type, S, X, T, r, v, d = 0, f = 0):
    """ Calculates the delta of an european vanilla option using the Generalized
        Black-Scholes-Merton formula

    Parameters:
        type (string): 'c' for a Call Option or 'p' for a Put Option
        S (float): Current(spot) price for the underlying Stock
        X (float): The option's strike value
        T (int): Weekdays until expiration date
        r (float): Risk-Free interest rate for expiration date
        v (float): Implied volatility
        d (float): Annualized dividend yield for underlying stock
        f (float): Foreign currency risk-free interest rate (for corrency options)
    
    Returns:
        delta (float): calculated delta for the closed-form Black-Scholes model
    """

    d1 = (math.log(S/X) + (r - f - d + v**2 / 2.0) * T) / (v * math.sqrt(T))
    
    if type == 'c':
        delta = math.exp((-d -f) * T) * norm.cdf(d1)
    else:
        delta = math.exp((-d -f) * T) * (norm.cdf(d1) - 1)
    return(delta)

def BSMgamma(type, S, X, T, r, v, d = 0, f = 0):
    """ Calculates the gamma of an european vanilla option using the Generalized
        Black-Scholes-Merton formula

    Parameters:
        type (string): 'c' for a Call Option or 'p' for a Put Option
        S (float): Current(spot) price for the underlying Stock
        X (float): The option's strike value
        T (int): Weekdays until expiration date
        r (float): Risk-Free interest rate for expiration date
        v (float): Implied volatility
        d (float): Annualized dividend yield for underlying stock
        f (float): Foreign currency risk-free interest rate (for corrency options)
    
    Returns:
        gamma (float): calculated gamma for the closed-form Black-Scholes model
    """

    d1 = (math.log(S/X) + (r - f - d + v**2 / 2.0) * T) / (v * math.sqrt(T))
    
    gamma = math.exp((-d -f) * T) * norm.ppf(d1) / (S * v * math.sqrt(T))
    return(gamma)

def BSMvega(type, S, X, T, r, v, d = 0, f = 0):
    """ Calculates the vega of an european vanilla option using the Generalized
        Black-Scholes-Merton formula

    Parameters:
        type (string): 'c' for a Call Option or 'p' for a Put Option
        S (float): Current(spot) price for the underlying Stock
        X (float): The option's strike value
        T (int): Weekdays until expiration date
        r (float): Risk-Free interest rate for expiration date
        v (float): Implied volatility
        d (float): Annualized dividend yield for underlying stock
        f (float): Foreign currency risk-free interest rate (for corrency options)
    
    Returns:
        vega (float): calculated vega for the closed-form Black-Scholes model
    """

    d1 = (math.log(S/X) + (r - f - d + v**2 / 2.0) * T) / (v * math.sqrt(T))
    
    vega = S * math.exp((-d -f) * T) * norm.ppf(d1) * math.sqrt(T)
    return(vega)

def BSMtheta(type, S, X, T, r, v, d = 0, f = 0):
    """ Calculates the theta of an european vanilla option using the Generalized
        Black-Scholes-Merton formula

    Parameters:
        type (string): 'c' for a Call Option or 'p' for a Put Option
        S (float): Current(spot) price for the underlying Stock
        X (float): The option's strike value
        T (int): Weekdays until expiration date
        r (float): Risk-Free interest rate for expiration date
        v (float): Implied volatility
        d (float): Annualized dividend yield for underlying stock
        f (float): Foreign currency risk-free interest rate (for corrency options)
    
    Returns:
        theta (float): calculated theta for the closed-form Black-Scholes model
    """

    d1 = (math.log(S/X) + (r - f - d + v**2 / 2.0) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    
    if type == 'c':
        theta = -S * math.exp((-d -f) * T) * norm.ppf(d1) * v / (2*math.sqrt(T)) - \
            (-d -f) * S * math.exp((-d -f) * T) * norm.cdf(d1) - \
                r * X * math.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -S * math.exp((-d -f) * T) * norm.ppf(d1) * v / (2*math.sqrt(T)) + \
            (-d -f) * S * math.exp((-d -f) * T) * norm.cdf(-d1) + \
                r * X * math.exp(-r * T) * norm.cdf(-d2)
    return(theta)

def BSMtheta_driftless(type, S, X, T, r, v, d = 0, f = 0):
    """ Calculates the driftless theta of an european vanilla option using the Generalized
        Black-Scholes-Merton formula

    Parameters:
        type (string): 'c' for a Call Option or 'p' for a Put Option
        S (float): Current(spot) price for the underlying Stock
        X (float): The option's strike value
        T (int): Weekdays until expiration date
        r (float): Risk-Free interest rate for expiration date
        v (float): Implied volatility
        d (float): Annualized dividend yield for underlying stock
        f (float): Foreign currency risk-free interest rate (for corrency options)
    
    Returns:
        theta (float): calculated driftless theta for the closed-form Black-Scholes model
    """

    d1 = (math.log(S/X) + (r - f - d + v**2 / 2.0) * T) / (v * math.sqrt(T))

    theta = -S * norm.ppf(d1) * v / (2 * math.sqrt(T))
    return(theta)

def BSMrho(type, S, X, T, r, v, d = 0, f = 0):
    """ Calculates the rho of an european vanilla option using the Generalized
        Black-Scholes-Merton formula

    Parameters:
        type (string): 'c' for a Call Option or 'p' for a Put Option
        S (float): Current(spot) price for the underlying Stock
        X (float): The option's strike value
        T (int): Weekdays until expiration date
        r (float): Risk-Free interest rate for expiration date
        v (float): Implied volatility
        d (float): Annualized dividend yield for underlying stock
        f (float): Foreign currency risk-free interest rate (for corrency options)
    
    Returns:
        rho (float): calculated rho for the closed-form Black-Scholes model
    """

    d1 = (math.log(S/X) + (r - f - d + v**2 / 2.0) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)

    if type == 'c':
        theta = T * X * math.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -T * X * math.exp(-r * T) * norm.cdf(-d2)
    return(theta)