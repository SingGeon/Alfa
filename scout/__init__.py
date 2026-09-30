"""Scout AI: scans a universe of crypto/stock assets for underdog opportunities."""
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

YFINANCE_RETRY_ATTEMPTS = 3
YFINANCE_RETRY_DELAY_SECONDS = 1.5


def _reset_yfinance_session() -> None:
    """Discard yfinance's process-wide session singleton (yfinance.data.YfData)
    so the next call rebuilds it from scratch.

    yfinance caches that singleton (cookie + negotiated "crumb" + the
    underlying curl_cffi session) for the life of the process. Verified
    directly: once its crumb negotiation fails once in a long-running
    process, every *subsequent* call keeps failing the exact same way
    (three unrelated tickers, several retries each, all rejected as
    "no price data") while a brand-new short-lived process succeeds
    against the very same ticker instantly - the state stuck inside
    *this* process's singleton is the actual problem, not Yahoo itself.
    Popping it from yfinance's internal registry forces a fresh session
    (and a fresh crumb negotiation) on the next call that constructs one -
    which must be a new yf.Ticker()/yf.screen() call, since an
    already-constructed Ticker keeps its own reference to the now-discarded
    instance (see _fetch_stock's comment).
    """
    try:
        from yfinance.data import SingletonMeta, YfData

        SingletonMeta._instances.pop(YfData, None)
    except Exception:
        logger.debug("Could not reset yfinance session singleton", exc_info=True)


def retry_yfinance(fn: Callable[[], T], is_empty: Callable[[T], bool], label: str) -> T:
    """Retry a yfinance call a few times before accepting an empty result.

    yfinance negotiates a "crumb" with Yahoo via curl_cffi's browser
    impersonation before most requests; that negotiation intermittently
    fails (curl_cffi's own ImpersonateError, or Yahoo rate-limiting the
    crumb endpoint itself), and yfinance's response is to silently retry
    the *actual* request without a crumb - which Yahoo then just as
    intermittently rejects as if the ticker had no data at all, regardless
    of which ticker it actually is. Verified directly: three unrelated
    tickers all failed identically within the same minute, then one of
    them alone succeeded seconds later in a fresh process - a transient
    data-source hiccup, not anything about those specific assets. A short
    bounded retry rides out that window instead of the caller surfacing a
    false "insufficient data"/"delisted" result for a perfectly fine asset.

    Also resets yfinance's session singleton between attempts (see
    _reset_yfinance_session) - a plain retry alone wasn't enough once the
    negotiation got stuck in a long-running process: it kept failing the
    same way on every attempt until the underlying session was discarded.
    """
    result = None
    for attempt in range(YFINANCE_RETRY_ATTEMPTS):
        try:
            result = fn()
        except Exception:
            # Logged (not just swallowed) so a genuine contract break - e.g.
            # yfinance/Yahoo changing a response shape - shows up somewhere
            # instead of looking identical to an ordinary transient crumb
            # failure that silently retries and resolves on its own.
            logger.debug("yfinance call (%s) raised, treating as empty for this attempt", label, exc_info=True)
            result = None
        if not is_empty(result):
            return result
        if attempt < YFINANCE_RETRY_ATTEMPTS - 1:
            logger.info("yfinance call (%s) came back empty, resetting session and retrying...", label)
            _reset_yfinance_session()
            time.sleep(YFINANCE_RETRY_DELAY_SECONDS)
    return result
