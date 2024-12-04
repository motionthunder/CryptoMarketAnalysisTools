import ccxt.async_support as ccxt  # Используем асинхронную версию ccxt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import logging
import warnings
import sys

# Отключение предупреждений
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)

@dataclass
class MarketData:
    timestamp: datetime
    price: float
    volume: float
    bid_volume: float
    ask_volume: float
    market_depth: float
    trades_count: int
    vwap: float
    orderbook: Dict
    btc_correlation: float
    relative_strength: float

class AdvancedMarketAnalyzer:
    def __init__(self,
                 exchange_name: str = 'binance',
                 target_profit: float = 0.005,
                 min_probability: float = 0.55,
                 max_positions: int = 5):

        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
        })
        self.target_profit = target_profit
        self.min_probability = min_probability
        self.max_positions = max_positions

        # Многоуровневые буферы для разных таймфреймов
        self.buffers = {
            'micro': {},  # 1 минута
            'small': {},  # 5 минут
            'medium': {},  # 15 минут
            'large': {}  # 1 час
        }

        # Максимальные размеры буферов
        self.buffer_sizes = {
            'micro': 100,
            'small': 15,
            'medium': 12,
            'large': 24
        }

        # Буферы для BTC данных
        self.btc_buffers = {
            timeframe: deque(maxlen=self.buffer_sizes[timeframe])
            for timeframe in self.buffers.keys()
        }

        # Активные позиции и история
        self.active_positions = {}
        self.trade_history = []

        self.min_requirements = {
            'volume_ratio': 1.2,  # Было 2.0
            'depth_ratio': 3.0,  # Было 5.0
            'bid_ask_imbalance': 1.2,  # Было 1.5
            'tick_size_ratio': 0.2,  # Было 0.1
            'min_btc_correlation': 0.2,  # Было 0.3
            'min_relative_strength': 1.1  # Было 1.2
        }

        # Веса для разных компонентов анализа
        self.weights = {
            'micro_structure': 0.3,
            'order_flow': 0.25,
            'relative_strength': 0.2,
            'market_depth': 0.15,
            'btc_correlation': 0.1
        }

    async def calculate_vwap(self, trades: List[Dict]) -> float:
        """Расчет VWAP на основе торгов"""
        if not trades:
            return 0

        total_volume = sum(trade['amount'] for trade in trades)
        if total_volume == 0:
            return 0

        vwap = sum(trade['price'] * trade['amount'] for trade in trades) / total_volume
        return vwap

    async def calculate_order_flow_toxicity(self, trades: List[Dict]) -> float:
        """Расчет VPIN (Volume-synchronized Probability of Informed Trading)"""
        if len(trades) < 50:
            return 0

        # Разделяем trades на временные корзины
        bucket_size = len(trades) // 50
        buckets = []

        for i in range(0, len(trades), bucket_size):
            bucket_trades = trades[i:i + bucket_size]
            buy_volume = sum(t['amount'] for t in bucket_trades if t['side'] == 'buy')
            sell_volume = sum(t['amount'] for t in bucket_trades if t['side'] == 'sell')
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                buckets.append(abs(buy_volume - sell_volume) / total_volume)

        return np.mean(buckets) if buckets else 0

    async def analyze_order_clustering(self, orderbook: Dict) -> Dict:
        """Анализ кластеризации ордеров в стакане"""
        bids = np.array([[price, vol] for price, vol in orderbook['bids']])
        asks = np.array([[price, vol] for price, vol in orderbook['asks']])

        def find_clusters(orders):
            if len(orders) < 2:
                return {'clusters': 0, 'largest_cluster': 0}

            # Используем иерархическую кластеризацию
            try:
                linkage_matrix = linkage(orders, method='ward')
                clusters = pd.cut(linkage_matrix[:, 2], bins=5)
                largest_cluster = orders[clusters.value_counts().idxmax()].sum(axis=0)[1]

                return {
                    'clusters': len(clusters.unique()),
                    'largest_cluster': largest_cluster
                }
            except:
                return {'clusters': 0, 'largest_cluster': 0}

        bid_clusters = find_clusters(bids)
        ask_clusters = find_clusters(asks)

        return {
            'bid_clusters': bid_clusters,
            'ask_clusters': ask_clusters,
            'imbalance': bid_clusters['largest_cluster'] /
                         (ask_clusters['largest_cluster'] + 1e-10)
        }

    async def calculate_relative_strength(self,
                                          symbol_data: List[float],
                                          market_data: List[float]) -> float:
        """Расчет относительной силы актива"""
        try:
            if len(symbol_data) < 2 or len(market_data) < 2:
                return 0

            # Приведение массивов к одинаковой длине
            min_len = min(len(symbol_data), len(market_data))
            symbol_data = np.array(symbol_data[-min_len:])
            market_data = np.array(market_data[-min_len:])

            # Проверка на наличие нулевых значений
            if np.any(symbol_data == 0) or np.any(market_data == 0):
                return 0

            symbol_returns = np.diff(symbol_data) / symbol_data[:-1]
            market_returns = np.diff(market_data) / market_data[:-1]

            # Проверка на наличие валидных данных
            if len(symbol_returns) == 0 or len(market_returns) == 0:
                return 0

            market_variance = np.var(market_returns)
            if market_variance == 0:
                return 0

            # Использование np.cov с проверкой размерности
            covariance_matrix = np.cov(symbol_returns, market_returns)
            if covariance_matrix.size < 4:
                return 0

            beta = covariance_matrix[0, 1] / market_variance
            alpha = np.mean(symbol_returns) - beta * np.mean(market_returns)

            return float(alpha / (beta + 1e-10))

        except Exception as e:
            logging.warning(f"Ошибка в расчете относительной силы: {e}")
            return 0

    async def calculate_btc_correlation(self,
                                        symbol_prices: List[float],
                                        btc_prices: List[float]) -> float:
        """Расчет корреляции с BTC"""
        if len(symbol_prices) < 2 or len(btc_prices) < 2:
            return 0

        # Приведение массивов к одинаковой длине
        min_len = min(len(symbol_prices), len(btc_prices))
        symbol_prices = symbol_prices[-min_len:]
        btc_prices = btc_prices[-min_len:]

        symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
        btc_returns = np.diff(btc_prices) / btc_prices[:-1]

        try:
            correlation = np.corrcoef(symbol_returns, btc_returns)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        except:
            return 0

    async def analyze_microstructure(self, symbol: str, market_data: MarketData) -> Dict:
        """Расширенный анализ микроструктуры рынка"""
        try:
            micro_buffer = self.buffers['micro'][symbol]

            if len(micro_buffer) < self.buffer_sizes['micro']:
                return {'score': 0, 'suitable': False}

            # 1. Анализ потока ордеров
            try:
                trades = await self.exchange.fetch_trades(symbol, limit=100)
                trades = [t for t in trades if t.get('price', 0) > 0 and t.get('amount', 0) > 0]

                if not trades:
                    return {'score': 0, 'suitable': False}

                vpin = await self.calculate_order_flow_toxicity(trades)
            except Exception as e:
                logging.warning(f"Ошибка при анализе потока ордеров {symbol}: {e}")
                vpin = 0

            # 2. Анализ кластеризации ордеров
            try:
                clusters = await self.analyze_order_clustering(market_data.orderbook)
            except Exception as e:
                logging.warning(f"Ошибка при анализе кластеризации {symbol}: {e}")
                clusters = {'imbalance': 1.0}

            # 3. Расчет метрик микроструктуры
            try:
                price_impact = abs(
                    market_data.vwap - market_data.price) / market_data.price if market_data.price != 0 else 1.0

                ask_price = market_data.orderbook['asks'][0][0]
                bid_price = market_data.orderbook['bids'][0][0]
                spread = (ask_price - bid_price) / market_data.price if market_data.price != 0 else 1.0
            except (ZeroDivisionError, IndexError) as e:
                logging.warning(f"Ошибка при расчете метрик {symbol}: {e}")
                price_impact = 1.0
                spread = 1.0

            # 4. Анализ объемов
            try:
                volumes = [data.volume for data in micro_buffer if data.volume > 0]
                if volumes:
                    volume_avg = np.mean(volumes)
                    volume_std = np.std(volumes)
                else:
                    volume_avg = volume_std = 0

                volume_profile = {
                    'current': market_data.volume,
                    'average': volume_avg,
                    'std': volume_std
                }
            except Exception as e:
                logging.warning(f"Ошибка при анализе объемов {symbol}: {e}")
                volume_profile = {
                    'current': 0,
                    'average': 0,
                    'std': 0
                }

            # 5. Оценка компонентов
            try:
                components = {
                    'order_flow': 1 - min(1.0, max(0.0, vpin)),
                    'clustering': min(1.0, clusters.get('imbalance', 1.0) if clusters.get('imbalance', 1.0) < 2 else 2 - clusters.get('imbalance', 1.0)),
                    'price_impact': 1 - min(1.0, price_impact * 100),
                    'spread_efficiency': 1 - min(1.0, spread * 100),
                    'volume_stability': 1 - min(1.0, volume_profile['std'] / (volume_profile['average'] + 1e-10))
                }
            except Exception as e:
                logging.warning(f"Ошибка при расчете компонентов {symbol}: {e}")
                return {'score': 0, 'suitable': False}

            # Веса компонентов микроструктуры
            micro_weights = {
                'order_flow': 0.3,
                'clustering': 0.25,
                'price_impact': 0.2,
                'spread_efficiency': 0.15,
                'volume_stability': 0.1
            }

            # Итоговая оценка микроструктуры
            micro_score = sum(score * micro_weights[component]
                              for component, score in components.items())

            return {
                'score': micro_score,
                'components': components,
                'volume_profile': volume_profile,
                'vpin': vpin,
                'clusters': clusters,
                'suitable': micro_score > 0.5
            }

        except Exception as e:
            logging.error(f"Критическая ошибка при анализе микроструктуры {symbol}: {e}")
            return {'score': 0, 'suitable': False}

    async def analyze_market_conditions(self,
                                        symbol: str,
                                        market_data: MarketData) -> Dict:
        """Комплексный анализ рыночных условий"""
        # 1. Микроструктурный анализ
        micro_analysis = await self.analyze_microstructure(symbol, market_data)

        # 2. Относительная сила
        relative_strength = market_data.relative_strength

        # 3. Корреляция с BTC
        btc_correlation = market_data.btc_correlation

        # 4. Анализ глубины рынка
        depth_ratio = market_data.market_depth / (market_data.volume + 1e-10)

        # Проверка минимальных требований
        requirements_met = {
            'volume': market_data.volume > self.min_requirements['volume_ratio'] * \
                      np.mean([d.volume for d in self.buffers['micro'][symbol]]),
            'depth': depth_ratio > self.min_requirements['depth_ratio'],
            'correlation': btc_correlation > self.min_requirements['min_btc_correlation'],
            'strength': relative_strength > self.min_requirements['min_relative_strength']
        }

        # Если не все требования выполнены, возвращаем отрицательный результат
        if not all(requirements_met.values()):
            return {
                'probability': 0,
                'suitable': False,
                'reason': f"Not met: {[k for k, v in requirements_met.items() if not v]}"
            }

        # Расчет итоговой вероятности
        components = {
            'micro_structure': micro_analysis['score'],
            'order_flow': 1 - micro_analysis['vpin'],
            'relative_strength': min(1.0, relative_strength / 2),
            'market_depth': min(1.0, depth_ratio / self.min_requirements['depth_ratio']),
            'btc_correlation': min(1.0, btc_correlation)
        }

        probability = sum(score * self.weights[component]
                          for component, score in components.items())

        return {
            'probability': probability,
            'suitable': probability > self.min_probability,
            'components': components,
            'micro_analysis': micro_analysis,
            'requirements': requirements_met
        }

    async def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Получение расширенных рыночных данных"""
        try:
            # Параллельный запрос всех необходимых данных
            ticker_task = asyncio.create_task(
                self.exchange.fetch_ticker(symbol)
            )
            orderbook_task = asyncio.create_task(
                self.exchange.fetch_order_book(symbol)
            )
            trades_task = asyncio.create_task(
                self.exchange.fetch_trades(symbol, limit=100)
            )

            # Дожидаемся выполнения всех запросов
            ticker, orderbook, trades = await asyncio.gather(
                ticker_task, orderbook_task, trades_task
            )

            # Проверка и обработка trades
            trades_data = [t['price'] for t in trades if 'price' in t]
            if len(trades_data) < 2:
                return None

            # Расчет VWAP
            vwap = await self.calculate_vwap(trades)

            # Расчет корреляции с BTC
            btc_prices = list(self.btc_buffers['micro'])
            btc_correlation = await self.calculate_btc_correlation(
                trades_data,
                btc_prices
            )

            # Расчет относительной силы
            relative_strength = await self.calculate_relative_strength(
                trades_data,
                btc_prices
            )

            return MarketData(
                timestamp=datetime.now(),
                price=ticker['last'],
                volume=ticker['quoteVolume'],
                bid_volume=sum(bid[1] for bid in orderbook['bids'][:5]),
                ask_volume=sum(ask[1] for ask in orderbook['asks'][:5]),
                market_depth=sum(b[1] for b in orderbook['bids'][:5]) +
                             sum(a[1] for a in orderbook['asks'][:5]),
                trades_count=len(trades),
                vwap=vwap,
                orderbook=orderbook,
                btc_correlation=btc_correlation,
                relative_strength=relative_strength
            )

        except Exception as e:
            logging.error(f"Ошибка получения данных для {symbol}: {e}")
            return None

    async def run_realtime_analysis(self):
        """Основной цикл анализа в реальном времени"""
        logging.info("Запуск расширенного анализа в реальном времени...")
        iteration = 0
        total_suitable_found = 0

        while True:
            try:
                iteration += 1
                logging.info(f"\nИтерация {iteration} ==================")

                # Получение списка активных пар
                logging.info("Загрузка рыночных данных...")
                markets = await self.exchange.fetch_tickers()

                # Фильтрация и сортировка по объему
                usdt_pairs = [(symbol, data['quoteVolume'])
                              for symbol, data in markets.items()
                              if symbol.endswith('/USDT') and data.get('quoteVolume') is not None]

                logging.info(f"Найдено {len(usdt_pairs)} USDT пар")

                top_pairs = sorted(
                    usdt_pairs,
                    key=lambda x: x[1],
                    reverse=True
                )[:500]

                symbols = [pair[0] for pair in top_pairs]
                logging.info(f"Топ-50 пар по объему: {', '.join(symbols[:5])}...")

                # Получение данных BTC для корреляции
                logging.info("Получение данных BTC...")
                btc_data = await self.fetch_market_data('BTC/USDT')
                if btc_data:
                    for timeframe in self.btc_buffers:
                        self.btc_buffers[timeframe].append(btc_data.price)
                    logging.info(f"Текущая цена BTC: {btc_data.price:.2f}")

                # Анализ каждой пары
                analyzed_count = 0
                suitable_pairs_this_iteration = 0

                for symbol in symbols:
                    analyzed_count += 1

                    if symbol in self.active_positions:
                        continue

                    try:
                        # Получение рыночных данных
                        market_data = await self.fetch_market_data(symbol)
                        if not market_data:
                            continue

                        # Обновление буферов
                        for timeframe in self.buffers:
                            if symbol not in self.buffers[timeframe]:
                                self.buffers[timeframe][symbol] = deque(maxlen=self.buffer_sizes[timeframe])
                            self.buffers[timeframe][symbol].append(market_data)

                        # Комплексный анализ
                        analysis = await self.analyze_market_conditions(symbol, market_data)

                        if analysis['suitable']:
                            suitable_pairs_this_iteration += 1
                            total_suitable_found += 1

                            logging.info(f"\n{'=' * 50}")
                            logging.info(f"НАЙДЕНА ПЕРСПЕКТИВНАЯ ВОЗМОЖНОСТЬ #{total_suitable_found}")
                            logging.info(f"Пара: {symbol}")
                            logging.info(f"Текущая цена: {market_data.price:.8f}")
                            logging.info(f"Объем (24ч): {market_data.volume:.2f}")
                            logging.info(f"Вероятность успеха: {analysis['probability']:.2%}")

                            logging.info("\nКомпоненты анализа:")
                            for component, value in analysis['components'].items():
                                logging.info(f"- {component}: {value:.3f}")

                            logging.info("\nМикроструктурный анализ:")
                            micro = analysis['micro_analysis']
                            for key, value in micro['components'].items():
                                logging.info(f"- {key}: {value:.3f}")

                            # Проверка возможности входа
                            if len(self.active_positions) < self.max_positions:
                                position = {
                                    'symbol': symbol,
                                    'entry_time': datetime.now(),
                                    'entry_price': market_data.price,
                                    'target_price': market_data.price * (1 + self.target_profit),
                                    'analysis': analysis,
                                    'volume': market_data.volume,
                                    'entry_vwap': market_data.vwap
                                }

                                self.active_positions[symbol] = position
                                logging.info("\nОТКРЫТА НОВАЯ ПОЗИЦИЯ:")
                                logging.info(f"Символ: {symbol}")
                                logging.info(f"Цена входа: {position['entry_price']:.8f}")
                                logging.info(f"Целевая цена: {position['target_price']:.8f}")
                                logging.info(f"Ожидаемый профит: {self.target_profit:.2%}")
                                logging.info(f"{'=' * 50}\n")

                        if analyzed_count % 10 == 0:
                            logging.info(f"Проанализировано {analyzed_count}/{len(symbols)} пар...")

                    except Exception as e:
                        logging.error(f"Ошибка при анализе {symbol}: {e}")

                logging.info(f"\nИтог итерации {iteration}:")
                logging.info(f"Проанализировано пар: {analyzed_count}")
                logging.info(f"Найдено подходящих: {suitable_pairs_this_iteration}")
                logging.info(f"Всего найдено с начала работы: {total_suitable_found}")
                logging.info(f"Активных позиций: {len(self.active_positions)}")

                # Мониторинг активных позиций
                if self.active_positions:
                    logging.info("\nМониторинг активных позиций...")
                    await self.monitor_positions()

                # Сохранение статистики
                self.save_statistics()

                await asyncio.sleep(0.5)  # Уменьшенная пауза между итерациями

            except Exception as e:
                logging.error(f"Ошибка в основном цикле: {e}")
                await asyncio.sleep(1)

    async def monitor_positions(self):
        """Расширенный мониторинг позиций"""
        for symbol, position in list(self.active_positions.items()):
            try:
                # Получение текущих данных
                market_data = await self.fetch_market_data(symbol)
                if not market_data:
                    continue

                current_price = market_data.price
                time_in_position = (datetime.now() - position['entry_time']).total_seconds() / 60

                # Проверка достижения профита
                if current_price >= position['target_price']:
                    profit_pct = (current_price - position['entry_price']) / position['entry_price']

                    trade_result = {
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(),
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit_pct': profit_pct,
                        'holding_time_minutes': time_in_position,
                        'entry_analysis': position['analysis']
                    }

                    self.trade_history.append(trade_result)
                    self.active_positions.pop(symbol)

                    logging.info(f"\nЗакрыта позиция {symbol}:")
                    logging.info(f"Прибыль: {profit_pct:.2%}")
                    logging.info(f"Время в позиции: {time_in_position:.1f} минут")

                else:
                    # Анализ текущего состояния позиции
                    current_analysis = await self.analyze_market_conditions(symbol, market_data)
                    logging.info(f"\nСтатус позиции {symbol}:")
                    logging.info(f"Время в позиции: {time_in_position:.1f} минут")
                    logging.info(
                        f"Текущая прибыль: {((current_price - position['entry_price']) / position['entry_price']):.2%}")
                    logging.info(f"Вероятность достижения цели: {current_analysis['probability']:.2%}")

            except Exception as e:
                logging.error(f"Ошибка при мониторинге позиции {symbol}: {e}")

    def save_statistics(self):
        """Сохранение расширенной статистики"""
        if not self.trade_history:
            return

        stats = {
            'total_trades': len(self.trade_history),
            'profitable_trades': len([t for t in self.trade_history if t['profit_pct'] > 0]),
            'avg_profit': np.mean([t['profit_pct'] for t in self.trade_history]),
            'avg_holding_time': np.mean([t['holding_time_minutes'] for t in self.trade_history]),
            'best_trade': max([t['profit_pct'] for t in self.trade_history]),
            'worst_trade': min([t['profit_pct'] for t in self.trade_history]),
            'current_positions': len(self.active_positions)
        }

        # Сохранение в файл
        with open('trading_statistics.txt', 'w') as f:
            for metric, value in stats.items():
                if 'profit' in metric:
                    f.write(f"{metric}: {value:.2%}\n")
                else:
                    f.write(f"{metric}: {value:.2f}\n")

        logging.info("\n=== Текущая статистика ===")
        for metric, value in stats.items():
            if 'profit' in metric:
                logging.info(f"{metric}: {value:.2%}")
            else:
                logging.info(f"{metric}: {value:.2f}")

    async def close(self):
        """Закрытие соединения с биржей"""
        await self.exchange.close()

async def main():
    try:
        analyzer = AdvancedMarketAnalyzer(
            target_profit=0.005,  # 0.5%
            min_probability=0.05,  # Минимальная вероятность для входа
            max_positions=5  # Максимум одновременных позиций
        )

        # Добавляем информацию о запуске
        logging.info("Запуск анализатора рынка...")
        logging.info(f"Целевая прибыль: {analyzer.target_profit:.2%}")
        logging.info(f"Минимальная вероятность: {analyzer.min_probability:.2%}")
        logging.info(f"Максимум позиций: {analyzer.max_positions}")

        await analyzer.run_realtime_analysis()

    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
    finally:
        await analyzer.close()
        logging.info("Завершение работы анализатора")

if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())