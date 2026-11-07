"""Position sizing and ATR-based risk levels for a single trade."""

import math
from typing import Literal


TradeDirection = Literal["long", "short"]


class RiskManager:
    """Calculates estimated risk limits; this module does not place orders."""

    REVOLUT_TARGET_RETURN = 0.045
    MINIMUM_ROUND_TRIP_COST_BPS = 20.0

    def __init__(
        self,
        risk_fraction: float = 0.015,
        atr_multiplier: float = 1.5,
        reward_risk_ratio: float = 2.0,
    ) -> None:
        if not math.isfinite(risk_fraction) or not 0.01 <= risk_fraction <= 0.02:
            raise ValueError("risk_fraction musi mieścić się w zakresie 0.01–0.02.")
        if not math.isfinite(atr_multiplier) or atr_multiplier <= 0:
            raise ValueError("atr_multiplier musi być większe od zera.")
        if not math.isfinite(reward_risk_ratio) or reward_risk_ratio < 2:
            raise ValueError("reward_risk_ratio musi wynosić co najmniej 2.0.")

        self.risk_fraction = risk_fraction
        self.atr_multiplier = atr_multiplier
        self.reward_risk_ratio = reward_risk_ratio

    def calculate_position(
        self,
        entry_price: float,
        atr: float,
        portfolio_capital: float,
        direction: TradeDirection = "long",
    ) -> dict[str, float | str]:
        """Return ATR stop/target and a spot-sized position for the given trade."""
        for name, value in (
            ("entry_price", entry_price),
            ("atr", atr),
            ("portfolio_capital", portfolio_capital),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} musi być skończoną wartością większą od zera.")
        if direction not in ("long", "short"):
            raise ValueError("direction musi mieć wartość 'long' albo 'short'.")

        stop_distance = self.atr_multiplier * atr
        if direction == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * self.reward_risk_ratio
            if stop_loss <= 0:
                raise ValueError(
                    "Stop-loss dla pozycji long wypada przy cenie zero lub niżej."
                )
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * self.reward_risk_ratio
            if take_profit <= 0:
                raise ValueError(
                    "Take-profit dla pozycji short wypada przy cenie zero lub niżej."
                )

        risk_budget = portfolio_capital * self.risk_fraction
        quantity_by_risk = risk_budget / stop_distance
        quantity_by_capital = portfolio_capital / entry_price
        position_size = min(quantity_by_risk, quantity_by_capital)
        estimated_risk = position_size * stop_distance

        return {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "stop_distance": stop_distance,
            "risk_reward_ratio": self.reward_risk_ratio,
            "risk_fraction": self.risk_fraction,
            "max_risk_amount": risk_budget,
            "estimated_risk_amount": estimated_risk,
            "position_size": position_size,
            "position_notional": position_size * entry_price,
            "capital_utilization": (position_size * entry_price) / portfolio_capital,
        }

    def calculate_revolut_position(
        self,
        purchase_price: float,
        current_price: float,
        atr: float,
        technical_signal: str,
        sentiment_score: float | None = None,
        resistance_price: float | None = None,
        rsi: float | None = None,
        macd: float | None = None,
        macd_signal: float | None = None,
        horizon_days: int = 3,
        bars_per_day: int = 6,
        is_mock: bool = False,
        round_trip_cost_bps: float = 40.0,
        adx: float | None = None,
        session_volume_ratio: float | None = None,
    ) -> dict[str, float | str | bool | None | dict[str, float | None]]:
        """Estimate a net +4.5% target and ATR stop for a 1–3 day horizon."""
        for name, value in (
            ("purchase_price", purchase_price),
            ("current_price", current_price),
            ("atr", atr),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} musi być skończoną wartością większą od zera.")
        if technical_signal not in {"KUP", "SPRZEDAJ", "CZEKAJ"}:
            raise ValueError("technical_signal musi mieć wartość KUP, SPRZEDAJ albo CZEKAJ.")
        if sentiment_score is not None and (
            not math.isfinite(sentiment_score) or not -1 <= sentiment_score <= 1
        ):
            raise ValueError("sentiment_score musi mieścić się w zakresie -1–1.")
        if (
            not math.isfinite(round_trip_cost_bps)
            or not self.MINIMUM_ROUND_TRIP_COST_BPS
            <= round_trip_cost_bps
            < 10_000
        ):
            raise ValueError(
                "Koszt round trip musi wynosić co najmniej 20 bps "
                "(10 bps przy zakupie i 10 bps przy sprzedaży) "
                "i być niższy niż 10 000 bps."
            )
        if adx is not None and (
            isinstance(adx, bool) or not math.isfinite(adx) or not 0 <= adx <= 100
        ):
            raise ValueError("ADX musi mieścić się w zakresie 0–100.")
        if session_volume_ratio is not None and (
            isinstance(session_volume_ratio, bool)
            or not math.isfinite(session_volume_ratio)
            or session_volume_ratio < 0
        ):
            raise ValueError(
                "session_volume_ratio musi być skończone i nieujemne."
            )
        for name, value in (
            ("resistance_price", resistance_price),
            ("rsi", rsi),
            ("macd", macd),
            ("macd_signal", macd_signal),
        ):
            if value is not None and (
                not math.isfinite(value)
                or (name in {"resistance_price", "rsi"} and value <= 0)
            ):
                raise ValueError(f"{name} ma nieprawidłową wartość.")
        if not isinstance(horizon_days, int) or not 1 <= horizon_days <= 3:
            raise ValueError("horizon_days musi być liczbą całkowitą z zakresu 1–3.")
        if not isinstance(bars_per_day, int) or bars_per_day <= 0:
            raise ValueError("bars_per_day musi być dodatnią liczbą całkowitą.")

        transaction_cost_fraction = round_trip_cost_bps / 20_000
        entry_cost_fraction = transaction_cost_fraction
        exit_cost_fraction = transaction_cost_fraction
        target_price = purchase_price * (
            (1 + entry_cost_fraction)
            * (1 + self.REVOLUT_TARGET_RETURN)
            / (1 - exit_cost_fraction)
        )
        stop_loss = purchase_price - self.atr_multiplier * atr
        if stop_loss <= 0:
            raise ValueError("Stop-Loss oparty na ATR wypada przy cenie zero lub niżej.")

        distance_to_target = max(0.0, target_price - current_price)
        horizon_volatility = atr * math.sqrt(horizon_days * bars_per_day)
        if distance_to_target == 0:
            volatility_probability = 100.0
        else:
            z_score = distance_to_target / horizon_volatility
            normal_cdf = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
            volatility_probability = max(0.0, min(100.0, 200 * (1 - normal_cdf)))

        technical_score = 50.0
        if technical_signal == "KUP":
            technical_score += 15
        elif technical_signal == "SPRZEDAJ":
            technical_score -= 20
        if rsi is not None:
            if 50 <= rsi <= 65:
                technical_score += 10
            elif rsi > 70:
                technical_score -= 15
            elif rsi < 45:
                technical_score -= 8
        if macd is not None and macd_signal is not None:
            technical_score += 10 if macd > macd_signal else -10
        if adx is not None and adx < 20:
            technical_score *= 0.5
        technical_score = max(0.0, min(100.0, technical_score))

        components: dict[str, float | None] = {
            "volatility_reachability": round(volatility_probability, 1),
            "technical": round(technical_score, 1),
            "news_sentiment": (
                None
                if sentiment_score is None
                else round(50 + sentiment_score * 50, 1)
            ),
            "resistance_penalty": 0.0,
            "adx_penalty": 0.0,
        }
        weighted_score = volatility_probability * 0.6 + technical_score * 0.3
        total_weight = 0.9
        if sentiment_score is not None:
            weighted_score += (50 + sentiment_score * 50) * 0.1
            total_weight += 0.1
        probability = weighted_score / total_weight

        if (
            resistance_price is not None
            and current_price < resistance_price < target_price
        ):
            resistance_penalty = 15.0 * (
                1 - (resistance_price - current_price) / distance_to_target
            )
            resistance_penalty = max(0.0, min(15.0, resistance_penalty))
            components["resistance_penalty"] = round(resistance_penalty, 1)
            probability -= resistance_penalty

        if adx is not None and adx < 20:
            components["adx_penalty"] = 25.0
            probability *= 0.5
        if is_mock:
            probability = min(probability, 20.0)
        probability = round(max(0.0, min(100.0, probability)), 1)
        target_reached = current_price >= target_price
        volume_kill_switch_status = (
            "unavailable"
            if session_volume_ratio is None
            else "blocked"
            if session_volume_ratio < 0.5
            else "passed"
        )
        adx_filter_passed = adx is not None and adx >= 20
        hard_filters_passed = (
            adx_filter_passed and volume_kill_switch_status == "passed"
        )
        if target_reached and not is_mock and hard_filters_passed:
            probability = 100.0

        return {
            "purchase_price": purchase_price,
            "current_price": current_price,
            "take_profit": target_price,
            "target_return_pct": self.REVOLUT_TARGET_RETURN * 100,
            "gross_target_return_pct": (
                (1 + entry_cost_fraction)
                * (1 + self.REVOLUT_TARGET_RETURN)
                / (1 - exit_cost_fraction)
                - 1
            ) * 100,
            "round_trip_cost_bps_assumption": round_trip_cost_bps,
            "entry_fee_bps_assumption": entry_cost_fraction * 10_000,
            "exit_fee_bps_assumption": exit_cost_fraction * 10_000,
            "estimated_round_trip_cost_pct": (
                entry_cost_fraction + exit_cost_fraction
            ) * 100,
            "net_target_return_pct": (
                target_price
                * (1 - exit_cost_fraction)
                / (purchase_price * (1 + entry_cost_fraction))
                - 1
            ) * 100,
            "stop_loss": stop_loss,
            "estimated_net_stop_return_pct": (
                (
                    stop_loss
                    * (1 - exit_cost_fraction)
                    / (purchase_price * (1 + entry_cost_fraction))
                    - 1
                )
                * 100
            ),
            "atr": atr,
            "atr_multiplier": self.atr_multiplier,
            "adx_14": adx,
            "adx_status": (
                "wstrzymany" if adx is not None and adx < 20
                else "trend" if adx is not None
                else "brak danych"
            ),
            "session_volume_ratio": session_volume_ratio,
            "volume_kill_switch_status": volume_kill_switch_status,
            "hard_filters_passed": hard_filters_passed,
            "hard_filter_status": (
                "zaliczone" if hard_filters_passed else "odrzucone"
            ),
            "horizon_days": horizon_days,
            "technical_signal": technical_signal,
            "resistance_price": resistance_price,
            "resistance_penalty": components["resistance_penalty"],
            "target_reached": target_reached,
            "success_probability_pct": probability,
            "probability_components": components,
            "is_mock": is_mock,
            "probability_note": (
                "Heurystyczna estymacja osiągalności celu na podstawie "
                "zmienności ATR, wskaźników technicznych, sentymentu i oporu. "
                "Nie jest skalibrowanym prawdopodobieństwem ani prognozą zysku."
            ),
        }

    def calculate_broker_filter_score(
        self,
        research: dict[str, object],
        round_trip_cost_bps: float = 40.0,
    ) -> dict[str, object]:
        """Score the five evidence-based broker filters and apply fail-closed gates."""
        if (
            not math.isfinite(round_trip_cost_bps)
            or not self.MINIMUM_ROUND_TRIP_COST_BPS
            <= round_trip_cost_bps
            < 10_000
        ):
            raise ValueError(
                "round_trip_cost_bps musi wynosić co najmniej 20 bps "
                "(10 bps przy zakupie i 10 bps przy sprzedaży) "
                "i być niższy niż 10 000 bps."
            )

        indicators = research.get("indicators")
        inputs = research.get("broker_inputs")
        price = research.get("price")
        if (
            not isinstance(indicators, dict)
            or not isinstance(inputs, dict)
            or not isinstance(price, (int, float))
            or not math.isfinite(price)
            or price <= 0
        ):
            raise ValueError("ResearchAgent nie dostarczył danych filtrów maklerskich.")

        technical_data = inputs.get("technical")
        if not isinstance(technical_data, dict):
            raise ValueError("Brak punktacji technicznej z ResearchAgenta.")
        technical_points = self._bounded_points(
            technical_data.get("points"),
            30.0,
            "technical points",
        )

        atr = indicators.get("atr_14")
        resistance = inputs.get("target_path")
        order_book = inputs.get("order_book")
        liquidity = inputs.get("liquidity")
        market_regime = inputs.get("market_regime")
        event_risk = inputs.get("event_risk")
        if not isinstance(atr, (int, float)) or not math.isfinite(atr) or atr <= 0:
            raise ValueError("Brak poprawnego ATR do oceny ryzyka.")
        if not all(
            isinstance(value, dict)
            for value in (resistance, order_book, liquidity, market_regime, event_risk)
        ):
            raise ValueError("Niekompletne wejściowe dane filtrów maklerskich.")
        adx_value = indicators.get("adx_14")
        adx_available = (
            isinstance(adx_value, (int, float))
            and not isinstance(adx_value, bool)
            and math.isfinite(adx_value)
            and 0 <= adx_value <= 100
        )
        adx_blocked = not adx_available or adx_value < 20

        cost_fraction = round_trip_cost_bps / 20_000
        entry_cost_fraction = cost_fraction
        exit_cost_fraction = cost_fraction
        target_price = price * (1 + entry_cost_fraction) * (
            1 + self.REVOLUT_TARGET_RETURN
        ) / (
            1 - exit_cost_fraction
        )
        stop_distance = self.atr_multiplier * atr
        stop_price = price - stop_distance
        if stop_price <= 0:
            raise ValueError("Stop-Loss oparty na ATR wypada przy cenie zero lub niżej.")
        entry_amount = price * (1 + entry_cost_fraction)
        risk_amount = entry_amount - stop_price * (1 - exit_cost_fraction)
        reward_amount = target_price * (1 - exit_cost_fraction) - entry_amount
        reward_risk_ratio = max(0.0, reward_amount) / risk_amount
        risk_reward_points = 15.0 if reward_risk_ratio >= 2.0 else 0.0
        cost_points = max(0.0, 10.0 * (1 - round_trip_cost_bps / 100))
        risk_points = risk_reward_points + cost_points

        reachability = self.calculate_revolut_position(
            purchase_price=price,
            current_price=price,
            atr=atr,
            technical_signal=str(research.get("signal", "CZEKAJ")),
            sentiment_score=(
                float(research["news"]["sentiment_score"])
                if isinstance(research.get("news"), dict)
                and research["news"].get("status") == "verified"
                else None
            ),
            resistance_price=(
                float(resistance["resistance_price"])
                if isinstance(resistance.get("resistance_price"), (int, float))
                else None
            ),
            rsi=float(indicators["rsi_14"]),
            macd=float(indicators["macd"]),
            macd_signal=float(indicators["macd_signal"]),
            horizon_days=3,
            is_mock=bool(research.get("is_mock")),
            round_trip_cost_bps=round_trip_cost_bps,
            adx=float(adx_value) if adx_available else None,
            session_volume_ratio=(
                float(liquidity["session_volume_ratio"])
                if isinstance(liquidity.get("session_volume_ratio"), (int, float))
                else None
            ),
        )
        target_probability = float(reachability["success_probability_pct"])
        reachability_points = 10.0 * target_probability / 100

        resistance_price = resistance.get("resistance_price")
        if isinstance(resistance_price, (int, float)):
            if resistance_price >= target_price:
                resistance_points = 5.0
            elif resistance_price > price:
                resistance_points = 5.0 * (
                    (resistance_price - price) / (target_price - price)
                )
            else:
                resistance_points = 0.0
        else:
            resistance_points = 0.0

        order_book_ratio = order_book.get("ask_bid_depth_ratio")
        largest_ask_wall_share = order_book.get("largest_ask_wall_share")
        if order_book.get("status") == "available" and isinstance(
            order_book_ratio, (int, float)
        ):
            order_book_points = (
                5.0
                if order_book_ratio <= 1.0
                else 3.0
                if order_book_ratio <= 1.5
                else 1.0
                if order_book_ratio <= 2.0
                else 0.0
            )
            if isinstance(largest_ask_wall_share, (int, float)):
                if largest_ask_wall_share >= 0.5:
                    order_book_points = 0.0
                elif largest_ask_wall_share >= 0.35:
                    order_book_points = min(order_book_points, 1.0)
                elif largest_ask_wall_share >= 0.25:
                    order_book_points = min(order_book_points, 3.0)
            else:
                order_book_points = 0.0
        else:
            order_book_points = 0.0
        target_points = reachability_points + resistance_points + order_book_points

        event_status = event_risk.get("status")
        events = event_risk.get("upcoming_events", [])
        high_event = event_status == "high_risk" or bool(events)
        event_complete = (
            event_risk.get("macro_calendar_status") == "available"
            and event_risk.get("corporate_calendar_status")
            in {"available", "not_applicable"}
        )
        event_points = 15.0 if event_complete and not high_event else 0.0

        volume_ratio = liquidity.get("session_volume_ratio")
        session_volume_ratio = volume_ratio
        if liquidity.get("status") == "available" and isinstance(
            volume_ratio, (int, float)
        ) and not isinstance(volume_ratio, bool):
            volume_points = (
                6.0
                if volume_ratio >= 1.5
                else 4.0
                if volume_ratio >= 1.0
                else 2.0
                if volume_ratio >= 0.7
                else 0.0
            )
        else:
            volume_points = 0.0
        regime = market_regime.get("regime")
        regime_points = (
            4.0
            if market_regime.get("status") == "available" and regime == "risk_on"
            else 2.0
            if market_regime.get("status") == "available"
            else 0.0
        )
        liquidity_points = volume_points + regime_points
        technical_components = technical_data.get("components")
        if not isinstance(technical_components, dict):
            technical_components = {}

        filters: dict[str, dict[str, object]] = {
            "technical_momentum": {
                "label": "Analiza techniczna i momentum",
                "points": round(technical_points, 1),
                "maximum": 30,
                "status": "available",
                "details": {
                    **technical_components,
                    "adx_14": adx_value,
                    "adx_threshold": 20.0,
                    "adx_status": (
                        "unavailable" if not adx_available
                        else "blocked" if adx_blocked
                        else "trend"
                    ),
                },
            },
            "risk_reward_costs": {
                "label": "Matematyka ryzyka, R/R i koszty",
                "points": round(risk_points, 1),
                "maximum": 25,
                "status": "available",
                "details": {
                    "risk_reward_ratio_net": round(reward_risk_ratio, 2),
                    "risk_reward_points": risk_reward_points,
                    "round_trip_cost_bps_assumption": round_trip_cost_bps,
                    "entry_fee_bps_assumption": entry_cost_fraction * 10_000,
                    "exit_fee_bps_assumption": exit_cost_fraction * 10_000,
                    "net_target_return_pct": self.REVOLUT_TARGET_RETURN * 100,
                    "gross_target_return_pct": (
                        target_price / price - 1
                    ) * 100,
                    "gross_target_price": target_price,
                    "cost_points": round(cost_points, 1),
                    "cost_note": (
                        "Uwzględniono co najmniej po 10 bps prowizji przy zakupie "
                        "i sprzedaży; rzeczywisty spread może zwiększyć koszt."
                    ),
                },
            },
            "target_resistance_order_book": {
                "label": "Cel +4,5% i przestrzeń do oporów",
                "points": round(target_points, 1),
                "maximum": 20,
                "status": (
                    "available"
                    if order_book.get("status") == "available"
                    and resistance.get("resistance_status") == "available"
                    else "partial"
                ),
                "details": {
                    "target_probability_heuristic_pct": target_probability,
                    "reachability_points": round(reachability_points, 1),
                    "resistance_price": resistance_price,
                    "resistance_points": round(resistance_points, 1),
                    "order_book_status": order_book.get("status"),
                    "ask_bid_depth_ratio": order_book_ratio,
                    "largest_ask_wall_share": largest_ask_wall_share,
                    "order_book_note": order_book.get("note")
                    or order_book.get("reason"),
                    "order_book_points": round(order_book_points, 1),
                },
            },
            "macro_event_risk": {
                "label": "Kalendarz makro i ryzyko wydarzeń",
                "points": round(event_points, 1),
                "maximum": 15,
                "status": "high_risk" if high_event else "verified" if event_complete else "incomplete",
                "details": {
                    "macro_calendar_status": event_risk.get("macro_calendar_status"),
                    "corporate_calendar_status": event_risk.get(
                        "corporate_calendar_status"
                    ),
                    "upcoming_events": events,
                    "reason": event_risk.get("reason"),
                },
            },
            "liquidity_volume_correlation": {
                "label": "Płynność, wolumen i korelacja międzyrynkowa",
                "points": round(liquidity_points, 1),
                "maximum": 10,
                "status": (
                    "available"
                    if liquidity.get("status") == "available"
                    and market_regime.get("status") == "available"
                    else "partial"
                ),
                "details": {
                    "volume_ratio": volume_ratio,
                    "session_volume_ratio": session_volume_ratio,
                    "session_volume_kill_switch": (
                        "unavailable"
                        if not isinstance(session_volume_ratio, (int, float))
                        else "blocked"
                        if session_volume_ratio < 0.5
                        else "passed"
                    ),
                    "latest_completed_session_volume": liquidity.get(
                        "latest_completed_session_volume"
                    ),
                    "average_prior_20_session_volume": liquidity.get(
                        "average_prior_20_session_volume"
                    ),
                    "volume_points": round(volume_points, 1),
                    "market_regime": regime,
                    "benchmark": market_regime.get("benchmark"),
                    "regime_points": round(regime_points, 1),
                },
            },
        }
        total_points = round(
            sum(float(item["points"]) for item in filters.values()),
            1,
        )
        disqualification_reasons: list[str] = []
        if adx_blocked:
            technical_details = dict(filters["technical_momentum"]["details"])
            technical_details["raw_points"] = round(technical_points, 1)
            filters["technical_momentum"]["details"] = technical_details
            filters["technical_momentum"]["points"] = round(
                technical_points * 0.25 if adx_available else 0.0,
                1,
            )
            filters["technical_momentum"]["status"] = (
                "blocked" if adx_available else "unavailable"
            )
            total_points = round(
                sum(float(item["points"]) for item in filters.values()),
                1,
            )
            disqualification_reasons.append(
                "ADX(14) poniżej 20; sygnał kierunkowy wstrzymany."
                if adx_available
                else "Brak ADX(14); filtr reżimu nie może zatwierdzić transakcji."
            )
        if (
            not isinstance(session_volume_ratio, (int, float))
            or isinstance(session_volume_ratio, bool)
            or not math.isfinite(session_volume_ratio)
        ):
            disqualification_reasons.append(
                "Brak ukończonej sesji i 20 sesji referencyjnych do filtra wolumenu."
            )
        elif session_volume_ratio < 0.5:
            disqualification_reasons.append(
                "Kill-switch: wolumen ostatniej ukończonej sesji jest poniżej 50% "
                "średniej z poprzednich 20 sesji."
            )
        if high_event:
            disqualification_reasons.append("Wykryto wydarzenie wysokiego ryzyka w horyzoncie.")
        if not event_complete:
            disqualification_reasons.append(
                "Brak kompletnego kalendarza makro; kwalifikacja zablokowana."
            )
        if research.get("is_mock") is not False:
            disqualification_reasons.append("Dane rynkowe są syntetyczne lub niepotwierdzone.")
        qualified = total_points > 75 and not disqualification_reasons
        if total_points <= 75:
            disqualification_reasons.append("Wynik nie przekracza 75 punktów.")
        return {
            "total_points": total_points,
            "maximum_points": 100,
            "qualification_threshold_exclusive": 75,
            "qualified": qualified,
            "filters": filters,
            "disqualification_reasons": disqualification_reasons,
            "probability_note": (
                "Punktacja filtrów jest regułowym modelem przesiewowym, "
                "a nie skalibrowanym prawdopodobieństwem zysku."
            ),
        }

    @staticmethod
    def _bounded_points(value: object, maximum: float, name: str) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or value > maximum
        ):
            raise ValueError(f"{name} musi mieścić się w zakresie 0–{maximum:g}.")
        return float(value)


def calculate_risk_parameters(
    current_price: float,
    atr: float,
    portfolio_capital: float,
    direction: TradeDirection = "long",
    risk_fraction: float = 0.015,
) -> dict[str, float | str]:
    """Convenience function returning a risk plan as a Python dictionary."""
    return RiskManager(risk_fraction=risk_fraction).calculate_position(
        entry_price=current_price,
        atr=atr,
        portfolio_capital=portfolio_capital,
        direction=direction,
    )
