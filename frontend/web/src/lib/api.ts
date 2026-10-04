// Typed client for the Flask API (api/routes.py + evaluation/api.py).
// Same-origin in production; proxied by Vite in dev.

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

type Params = Record<string, string | number | boolean | null | undefined>;

export function buildUrl(path: string, params?: Params): string {
  const url = new URL(path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
    }
  }
  return url.pathname + url.search;
}

async function request<T>(path: string, params?: Params, init?: RequestInit): Promise<T> {
  const resp = await fetch(buildUrl(path, params), init);
  const body = await resp.json().catch(() => null);
  if (!resp.ok) throw new ApiError((body && body.error) || `${resp.status} ${resp.statusText}`, resp.status);
  return body as T;
}

export const apiGet = <T,>(path: string, params?: Params) => request<T>(path, params);

// ---------------------------------------------------------------- types

export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface PredictionPoint {
  timestamp: string;
  predicted_price: number;
  lower: number;
  upper: number;
}

export interface PredictResponse {
  interval: string;
  steps: number;
  backend: string;
  model_variant: string;
  used_sentiment: boolean;
  sentiment_avg: number;
  sentiment_contribution_pct: number | null;
  trained_ago_seconds: number;
  confidence: number;
  last_known_price: number;
  predictions: PredictionPoint[];
}

export interface CurrentPrice {
  price: number;
  change_24h_pct?: number | null;
  volume_24h?: number | null;
  source?: string;
}

export interface OutlookLeg {
  interval: string;
  timestamp: string;
  last_known_price: number;
  predicted_price: number;
  lower: number;
  upper: number;
  change_pct: number;
  confidence: number;
}
export interface OutlookResponse {
  daily: OutlookLeg;
  weekly: OutlookLeg;
}

export interface NewsItem {
  title: string;
  url: string;
  source?: string;
  published_at?: string;
  sentiment?: { label?: string; compound?: number };
}

export interface AccuracyResponse {
  count: number;
  mape: number | null;
  points: { timestamp: string; predicted_price: number; actual_price: number; error_pct: number | null }[];
}

export interface SignalStat {
  count: number;
  correct?: number;
}
export interface SignalAccuracyResponse {
  buy?: SignalStat;
  sell?: SignalStat;
  wait?: SignalStat;
}

export interface L2Chain {
  name: string;
  display_name: string;
  tvl_usd: number;
  launch_year: number;
  tier: "established" | "growing" | "emerging";
}

export type PatternDirection = "bullish" | "bearish" | "neutral";
export interface CandlePattern {
  name: string;
  direction: PatternDirection;
  note: string;
  timestamp: string;
  index: number;
  target?: number | null;
  confirmed?: boolean;
}
export interface GeometricPattern {
  name: string;
  direction: PatternDirection;
  note: string;
  levels: { timestamp: string; price: number }[];
  upper_count?: number;
  neckline?: number | null;
  target?: number | null;
  confirmed?: boolean;
}
export interface PatternsResponse {
  candlestick: CandlePattern[];
  geometric: GeometricPattern[];
}

export type AssetType = "crypto" | "stock";

export interface ScoutResult {
  asset_type: AssetType;
  id: string;
  symbol: string;
  name: string;
  current_price: number;
  predicted_price: number;
  predicted_change_pct: number;
  confidence: number;
  sentiment_avg: number;
  articles_analyzed: number;
  score: number;
  tier?: "underdog" | "mega_cap" | string;
  scanned_at?: string;
}

export interface Pin {
  asset_type: AssetType;
  id: string;
  symbol: string;
  name: string;
}

export interface ScoutDetail {
  asset_type: AssetType;
  id: string;
  symbol: string;
  name: string;
  current_price: number;
  confidence: number;
  sentiment_avg: number;
  horizon_days: number;
  narrative: string;
  candles: Candle[];
  predictions: PredictionPoint[];
  news: NewsItem[];
}

export interface Quote {
  price: number;
  change_24h_pct?: number | null;
}

// ---- evaluation

export type EvalStatus = "pending" | "completed" | "expired";
export interface EvalRow {
  id: number;
  created_at: string;
  interval: string;
  model_name: string;
  horizon_steps: number;
  price_at_prediction: number;
  predicted_final_price: number;
  actual_final_price: number | null;
  pct_error: number | null;
  direction_correct: number | boolean | null;
  status: EvalStatus;
  snapshot_url: string;
  actual_path: unknown[] | string;
}
export interface EvalModelSummary {
  model_name: string;
  predictions: number;
  completed: number;
  pending: number;
  mean_abs_pct_error: number | null;
  direction_accuracy_pct: number | null;
}
export interface EvalPredictionsResponse {
  page: number;
  pages: number;
  total: number;
  summary: EvalModelSummary[];
  rows: EvalRow[];
}
export interface EvalStat {
  predictions: number;
  completed: number;
  pending: number;
  expired?: number;
  mean_abs_pct_error: number | null;
  mean_abs_error_usd?: number | null;
  mean_pct_error?: number | null;
  direction_accuracy_pct: number | null;
}
export interface EvalStatsResponse {
  overall: EvalStat | null;
  by_model: (EvalStat & { model: string })[];
  by_interval: (EvalStat & { interval: string })[];
  by_model_interval: (EvalStat & { model: string; interval: string })[];
  by_created_day: (EvalStat & { created_day: string })[];
}
export interface EvalDayDetail {
  predictions: number;
  mean_abs_pct_error: number | null;
  direction_accuracy_pct: number | null;
  best_model?: { model_name: string; mean_abs_pct_error: number | null } | null;
  worst_model?: { model_name: string; mean_abs_pct_error: number | null } | null;
  gif: boolean;
  pdf: boolean;
}

// ---- evolution (ml/evolution.py, the "tuned" population)

export interface AliveOrganism {
  id: number;
  age: number;
  generation: number;
  born: string;
  money: number;
  peak_money: number;
  position: number;
  trades: number;
  fees_paid: number;
  trading_pnl: number;
  wins: number;
  losses: number;
  combos: number;
  emotion: string;
  mood: number;
  energy: number;
  voting: boolean;
  inputs: string[];
  parents: number[];
  boldness: number;
  temperament: number;
  news_sensitivity: number;
  window: number;
  patience: number;
  ridge_alpha: number;
}

export interface DeathRow {
  id: number;
  born: string;
  died: string;
  age: number;
  generation: number;
  cause: string;
  money: number;
  peak_money: number;
  trades: number;
  fees_paid: number;
  tax_paid: number;
  wins: number;
  losses: number;
  combos: number;
  inputs: string[];
  parents: number[];
  emotion: string;
  earned_mood: number;
  price: number;
  move_24_pct: number;
  volatility_vs_month: number | null;
  market_sentiment: number | null;
  sentiment_source: string | null;
  growth_pct_per_100: number;
  killing_prediction: { h: number; predicted_pct: number; actual_pct: number } | null;
  last_48: { fees: number; taxes: number; trading_loss: number; worst_candle_pct: number } | null;
  boldness: number;
  temperament: number;
  news_sensitivity: number;
  window: number;
  patience: number;
  ridge_alpha: number;
}
export interface EvolutionDeathsResponse {
  page: number;
  page_size: number;
  pages: number;
  total: number;
  rows: DeathRow[];
}

export interface AncestorNode {
  id: number;
  level: number;
  generation: number;
  age: number;
  dead: boolean;
  cause?: string;
  inputs: string[];
  known: boolean;
}
export interface OrganismDetail extends AliveOrganism {
  dead: boolean;
  cause?: string;
  died?: string;
  price?: number;
  move_24_pct?: number;
}
export interface EvolutionOrganismResponse {
  organism: OrganismDetail;
  ancestors: AncestorNode[];
  children: { id: number; generation: number; age: number; dead: boolean }[];
}

export interface EvolutionTimelinePoint {
  time: string;
  price: number | null;
  deaths: number;
  alive: number;
  mood: number;
  fund: number | null;
  buy_hold: number | null;
  richest: number | null;
  boldness: number;
  temperament: number;
  news_sensitivity: number;
  window: number;
  patience: number;
  avg_generation: number;
  shares: Record<string, number>;
}
export interface EvolutionCause {
  cause: string;
  deaths: number;
  share_pct: number;
  avg_age: number | null;
}
export interface EvolutionGeneration {
  from: number;
  to: number;
  organisms: number;
  alive: number;
  avg_lifespan: number | null;
  causes: Record<string, number>;
  input_shares: Record<string, number>;
  boldness: number;
  temperament: number;
  news_sensitivity: number;
  window: number;
  patience: number;
}
export interface TrackRecordRow {
  h: number;
  n: number;
  skill: number | null;
  direction_pct: number | null;
  recent_n: number;
  recent_skill: number | null;
  recent_direction_pct: number | null;
}
export interface EvolutionSummary {
  alive: number;
  births: number;
  deaths: number;
  max_generation: number;
  avg_lifespan_of_dead: number | null;
  oldest_alive: number;
  emotion: string;
  mood: number;
  lived_from: string;
  lived_to: string;
  candles_lived: number;
  market_sentiment: { source: string; value: number } | null;
}
export interface EvolutionMoney {
  currency: string;
  start: number;
  since: string;
  fund: number;
  fund_return_pct: number;
  fund_peak: number;
  fund_trades: number;
  fund_fees: number;
  fund_position: number;
  buy_hold: number;
  buy_hold_return_pct: number;
  richest: { id: number; age: number; money: number } | null;
  total_alive_money: number;
  long: number;
  short: number;
  out: number;
}
export interface EvolutionReport {
  interval: string;
  genes: string[];
  summary: EvolutionSummary;
  money: EvolutionMoney | null;
  timeline: EvolutionTimelinePoint[];
  causes: EvolutionCause[];
  gene_survival: Record<string, { avg_lifespan: number; deaths: number }>;
  generations: EvolutionGeneration[];
  track_record: TrackRecordRow[];
  alive: AliveOrganism[];
}

// ---- strategy fund (ml/strategy_fund.py)

export interface FundNow {
  position: number;
  box: number;
  pieces: { core: number; box: number; range: number };
  vol_forecast_annual_pct: number | null;
  next_reselection: string;
}
export interface FundMoneyStat {
  money: number;
  ret_pct: number;
  cagr_pct: number;
  max_dd_pct: number;
  sharpe: number;
  trades: number;
  fees: number;
  exposure_pct: number;
}
export interface FundPeriod {
  fund: FundMoneyStat;
  buy_hold: FundMoneyStat;
}
export interface FundYear {
  year: number;
  fund_pct: number;
  fund_dd: number;
  buy_hold_pct: number;
  buy_hold_dd: number;
  trades: number;
}
export interface FundCurvePoint {
  time: string;
  fund: number;
  buy_hold: number;
  position: number;
}
export interface FundGenePoint {
  time: string;
  core: number;
  target: number;
  box_w: number;
  range_w: number;
  long_short: number;
}
export interface FundStrategyGenes {
  core: number;
  target: number;
  box_w: number;
  range_w: number;
  long_short: boolean;
  [key: string]: number | boolean;
}
export interface FundStrategy {
  genes: FundStrategyGenes;
  now: { core: number; box: number; range: number };
  position: number;
  score: number;
}
export interface FundSettings {
  population: number;
  top_k: number;
  replace: number;
  reselect_days: number;
  lookback_days: number;
  seeds: number[];
  fee: number;
  dd_penalty: number;
}
export interface FundReport {
  interval: string;
  price: number;
  last_candle: string;
  computed_at: string;
  trading_from: string;
  test_from: string;
  now: FundNow;
  whole_life: FundPeriod;
  test: FundPeriod;
  years: FundYear[];
  curve: FundCurvePoint[];
  genes_over_time: FundGenePoint[];
  strategies: FundStrategy[];
  settings: FundSettings;
}

// ---- pins

export async function addPin(pin: Pin): Promise<void> {
  await request("/api/scout/pins", undefined, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pin),
  });
}
export async function removePin(assetType: AssetType, id: string): Promise<void> {
  await request("/api/scout/pins", { asset_type: assetType, id }, { method: "DELETE" });
}
