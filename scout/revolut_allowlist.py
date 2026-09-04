"""Curated allowlist of assets actually tradeable on Revolut - the scout
only recommends things the user can actually go buy.

Revolut has no public catalog API for its retail brokerage/crypto product
(the separate "Revolut X Crypto Exchange" API is an institutional product
and is deliberately not used anywhere in this project). So this list is
maintained by hand instead of fetched live. Crypto availability in
particular varies by country and changes over time as Revolut adds/removes
coins - if a coin you can actually trade is missing (or one here has been
delisted), just edit REVOLUT_CRYPTO_SYMBOLS below; nothing else needs to change.
"""
from __future__ import annotations

# Symbols (uppercase, no exchange suffix) Revolut's crypto product supports
# for most markets. Deliberately conservative: an allowlist means anything
# NOT listed here is excluded from the scout, so it's safer to leave out an
# edge-case coin than to include one the user can't actually buy.
REVOLUT_CRYPTO_SYMBOLS: set[str] = {
    # majors / L1s
    "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "SOL", "DOT", "AVAX", "ATOM",
    "ALGO", "NEAR", "FTM", "HBAR", "EGLD", "XTZ", "TRX", "ETC", "ICP", "VET",
    "XLM", "FLOW", "ONE", "KSM", "MINA", "CELO", "ROSE", "GLMR", "ASTR",
    # DeFi / Ethereum ecosystem
    "UNI", "LINK", "AAVE", "MKR", "COMP", "SNX", "CRV", "SUSHI", "YFI",
    "BAT", "ZRX", "OMG", "KNC", "REN", "BAL", "1INCH", "UMA", "DYDX",
    "LDO", "GRT", "ENS", "LRC", "MASK", "BNT", "STORJ",
    # scaling / L2
    "MATIC", "POL", "OP", "ARB", "IMX",
    # newer L1s / infra
    "APT", "SUI", "INJ", "TIA", "SEI", "RNDR", "RENDER", "FET", "RUNE",
    "KAVA", "QNT", "STX", "WLD",
    # meme / high-attention
    "DOGE", "SHIB", "PEPE", "GMT", "APE", "JASMY", "BLUR",
    # gaming / metaverse
    "SAND", "MANA", "AXS", "CHZ", "ENJ", "GALA",
    # stablecoins
    "USDT", "USDC", "DAI",
    # other long-listed majors
    "THETA", "ANKR",
}

# yfinance screener `exchange` codes covering the US markets Revolut Trading
# supports for essentially all users (NASDAQ Global Select/Global/Capital
# Market, NYSE, NYSE American, NYSE Arca). Excludes OTC/pink-sheet codes
# (e.g. "PNK") and foreign exchanges Revolut doesn't universally list on -
# those aren't reliably buyable even when the ticker itself looks familiar.
REVOLUT_STOCK_EXCHANGES: set[str] = {"NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS"}
