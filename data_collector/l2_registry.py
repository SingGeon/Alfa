"""Curated list of well-known Ethereum Layer 2 rollups, editable by hand -
same pattern as scout/revolut_allowlist.py. DeFiLlama's /v2/chains endpoint
lists 400+ chains with no "is this an Ethereum L2" flag, so this is what
narrows that down to ones actually worth showing on an ETH dashboard.

`name` must match the chain `name` field DeFiLlama returns exactly (see
data_collector.defillama_client.get_l2_tvls) - check https://api.llama.fi/v2/chains
if a listed L2 stops matching after a DeFiLlama rename.
"""

ETHEREUM_L2S = [
    {"name": "Arbitrum", "display_name": "Arbitrum One", "launch_year": 2021},
    {"name": "OP Mainnet", "display_name": "OP Mainnet (Optimism)", "launch_year": 2021},
    {"name": "Base", "display_name": "Base", "launch_year": 2023},
    {"name": "ZKsync Era", "display_name": "ZKsync Era", "launch_year": 2023},
    {"name": "Starknet", "display_name": "Starknet", "launch_year": 2021},
    {"name": "Linea", "display_name": "Linea", "launch_year": 2023},
    {"name": "Scroll", "display_name": "Scroll", "launch_year": 2023},
    {"name": "Blast", "display_name": "Blast", "launch_year": 2024},
    {"name": "Mantle", "display_name": "Mantle", "launch_year": 2023},
    {"name": "Polygon zkEVM", "display_name": "Polygon zkEVM", "launch_year": 2023},
]
