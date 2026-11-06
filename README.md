# MAFSE

Prosty przepływ multi-agent korzystający z modelu Gemini:

1. `PlannerAgent` tworzy plan zadania.
2. `ResearchAgent` opracowuje ustalenia na podstawie planu.
3. `WriterAgent` przygotowuje odpowiedź końcową.

## Konfiguracja

Zainstaluj zależności:

```bash
python -m pip install -r requirements.txt
```

Ustaw klucz API w zmiennej środowiskowej `GEMINI_API_KEY` (nie zapisuj klucza
w kodzie ani repozytorium). Opcjonalnie możesz ustawić nazwę modelu przez
`GEMINI_MODEL`; domyślnie używany jest `gemini-3.8-flash`.

Przykład uruchomienia w macOS/Linux:

```bash
export GEMINI_API_KEY="twój-klucz-api"
python main.py "Przeanalizuj parę BTC/USDT" --capital 10000
```

`main.py` kolejno tworzy plan Gemini, pobiera dane Yahoo Finance dla wskazanych
interwałów, oblicza ryzyko dla obu kierunków i generuje raport. Wartość
`--capital` podaj w USD — dane cenowe Yahoo Finance dla BTC-USD są denominowane
w USD. Przy mapowaniu par `*/USDT` na `*-USD` nie jest wykonywana konwersja walut.
Błędy Gemini i Yahoo Finance HTTP 503 są ponawiane do trzech prób.
Wywołania Gemini w całej aplikacji korzystają ze wspólnego retry w
`GeminiClient`, z wykładniczym opóźnieniem; dotyczy ono wyłącznie HTTP 503.
`tenacity` jest zadeklarowane w `requirements.txt`, więc przy przygotowaniu
środowiska wystarczy standardowa instalacja zależności projektu.

Uruchom dashboard Streamlit:

```bash
streamlit run app.py
```

Po uruchomieniu dashboard skanuje BTC-USD, ETH-USD, SOL-USD, NVDA i AAPL na
interwale 4h. Kandydat do Top 3 musi mieć sygnał KUP, wynik ściśle powyżej
75/100 i przejść wszystkie blokady jakości danych. Wynik składa się z pięciu
filtrów: technika i momentum (30 pkt), R/R i koszty (25 pkt), cel +4,5% i
przestrzeń do oporu (20 pkt), kalendarz makro/event risk (15 pkt) oraz
płynność i korelacja międzyrynkowa (10 pkt). Karty i szczegółowy widok
pokazują punktację każdego filtra, status źródeł i uzasadnienie. Wyniki skanu
są cache’owane przez 15 minut; założony koszt round trip można zmienić w
panelu bocznym (domyślnie 40 bps, czyli 0,40%). To założenie kosztu, nie pobrana taryfa
Revoluta ani rzeczywisty spread.
Filtr ADX(14) wstrzymuje sygnały, gdy ADX < 20; dane ADX są również widoczne
w szczegółowej analizie i panelu pozycji. Twardy kill-switch odrzuca aktywo,
jeśli wolumen ostatniej ukończonej sesji jest mniejszy niż 50% średniego
wolumenu z poprzednich 20 sesji. Przy niepełnej historii filtr wolumenu
zachowuje się fail-closed.

Skan działa dwuetapowo: najpierw równolegle liczy świece, wskaźniki i ryzyko
w czystym Pythonie; dopiero trzy najlepsze aktywa z wynikiem technika+ryzyko
co najmniej 40/55 mogą przejść do pobrania wiadomości i review Gemini.
Review porównuje wcześniej skorelowane nagłówki i źródła; nie jest niezależnym
audytorem prawdziwości artykułów.
Wyniki skanu, analizy szczegółowej, pozycji Revolut i udanych odpowiedzi Gemini
są keszowane w Streamlit przez 15 minut. Połączenia sieciowe oraz wywołania
Gemini w dashboardzie mają timeout 4 s; przy timeout/HTTP 429 jakość wiadomości
jest pomijana i stosowana jest neutralna ocena sentymentu. Odpowiedzi błędne
lub ograniczone przez API nie są utrwalane w cache Gemini. Przycisk odświeżenia
skanu czyści też cache review Gemini.

Kalendarz makro nie ma obecnie dostępnego, kompletnego źródła. Dlatego
kwalifikacja do Top 3 jest celowo blokowana (fail-closed), dopóki kalendarz
nie będzie mógł potwierdzić braku istotnych wydarzeń; dashboard nie interpretuje
braku danych jako braku ryzyka. Dla obsługiwanych par krypto punktacja przestrzeni
do celu wykorzystuje migawkę Binance Spot: oferty ask przed celem +4,5%,
głębokość bid w pasie ±4,5% i udział największej ściany ask. Nie jest to
arkusz Revoluta ani akcje; dla pozostałych aktywów punkty order book są zerowe.
Przestrzeń do celu uwzględnia cel brutto +4,5% powiększony o założony koszt,
tak aby szacowany zwrot netto pozostał na poziomie +4,5%.
Szczegółową analizę można uruchomić dla dowolnego tickera
Yahoo Finance. Confidence Score WriterAgenta jest odrębną heurystyką, a nie
prawdopodobieństwem zysku; przy CZEKAJ wykres pokazuje scenariuszowe poziomy
long/short, a nie rekomendację.

Panel „Pozycja z Revoluta” przyjmuje ticker i cenę zakupu, a następnie wylicza
cenę realizacji celu brutto powiększoną o założone koszty round trip, tak aby
po kosztach pozostało +4,5% netto. Stop-Loss pozostaje 1,5 × ATR poniżej ceny
zakupu; dashboard pokazuje również szacowany wynik netto na tym poziomie oraz
status filtra ADX. Pokazywana szansa osiągnięcia celu w horyzoncie do 3 dni jest
heurystyczną estymacją z bieżącego ATR 4h, wskaźników, zweryfikowanego
sentymentu i lokalnego oporu; nie jest skalibrowanym prawdopodobieństwem
rynkowym. Dla danych syntetycznych kalkulacja jest blokowana.

Aby przetestować cały pipeline bez klucza API i bez połączenia z siecią,
uruchom symulację na deterministycznych danych:

```bash
python dry_run.py
```

Skrypt sprawdza planowanie, pobieranie i obliczanie wskaźników z mockowanych
świec, limity ryzyka oraz końcowy raport.

Aby sprawdzić połączenie z API Gemini, uruchom:

```bash
python test_api.py
```

W razie błędu 503 skrypt ponawia próbę do trzech razy z rosnącym opóźnieniem.

## Dane rynkowe

`research_agent.py` pobiera świece OHLCV z Yahoo Finance (domyślnie BTC-USD)
i oblicza RSI(14), MACD(12, 26, 9) oraz ATR(14) przy użyciu
`pandas-ta-classic` (API zgodne z pandas-ta), a także ADX(14). Do działania nie wymaga klucza API:

```bash
python research_agent.py --symbol BTC-USD --timeframe 1d --limit 100
```

Wyniki są wypisywane jako JSON ze świecami OHLCV, najnowszą ceną, RSI, MACD,
ATR, ADX i wstępnym sygnałem KUP/SPRZEDAJ/CZEKAJ; ADX < 20 wstrzymuje sygnał
kierunkowy. Filtr płynności porównuje ostatnią ukończoną sesję z 20 poprzednimi
sesjami i odrzuca relację wolumenu poniżej 0,5. Dla par `*/USDT` symbol Yahoo
Finance jest mapowany na `*-USD`; wartości Yahoo pozostają denominowane w USD.
Jeśli Yahoo Finance nie zwróci danych dla BTC-USD, agent generuje realistyczne,
syntetyczne świece testowe i oznacza wynik jako `is_mock: true`. Takie dane
służą wyłącznie do testów; końcowy WriterAgent wymusza dla nich sygnał `CZEKAJ`.

ResearchAgent pobiera też dostępne dane fundamentalne Yahoo Finance oraz
najnowsze nagłówki przez Google News RSS. Wiadomość jest uwzględniana tylko,
gdy pochodzi z dozwolonej listy domen wydawców (m.in. Reuters, AP, BBC,
Bloomberg, CNBC, FT, WSJ i Yahoo Finance) i podobny nagłówek pojawił się w
co najmniej dwóch różnych źródłach w ostatnich siedmiu dniach. Pozostałe
nagłówki, w tym oznaczone jako plotka, spekulacja lub informacja
niepotwierdzona, są pomijane. To konserwatywna weryfikacja źródła i zgodności relacji,
nie pełny fact-check ani dowód prawdziwości artykułu. Awaria RSS i brak danych
fundamentalnych są jawnie oznaczane; nie są zastępowane zmyślonymi danymi.

## Zarządzanie ryzykiem

`RiskManager` z `risk_manager.py` oblicza poziomy SL/TP na podstawie ATR i
rozmiar pozycji. Ryzyko można ustawić w zakresie 1–2% kapitału (domyślnie
1,5%), a stosunek zysku do ryzyka wynosi co najmniej 1:2. Wielkość pozycji jest dodatkowo ograniczona
do wartości kapitału (bez dźwigni):

```python
from risk_manager import RiskManager

plan = RiskManager().calculate_position(
    entry_price=100_000,
    atr=1_000,
    portfolio_capital=10_000,
    direction="long",
)
```

Wynik to słownik gotowy do serializacji do JSON. Wyliczone ryzyko jest
szacunkowe i nie uwzględnia prowizji, poślizgu cenowego ani luk rynkowych.

## Planowanie analizy

`PlannerAgent` z `planner_agent.py` używa klienta Gemini do utworzenia
ustrukturyzowanego planu dla ResearchAgenta i RiskManagera. Rozpoznaje parę
podaną w celu i normalizuje ją do tickera Yahoo Finance (np. `BTC/USD` →
`BTC-USD`), dobiera ramy czasowe do stylu transakcji i zwraca wskaźniki,
pytania badawcze oraz parametry ryzyka:

```python
from mafse import GeminiClient
from mafse.agent import AgentContext
from planner_agent import PlannerAgent

planner = PlannerAgent(GeminiClient.from_env())
plan = planner.generate_plan("Przeanalizuj BTC/USD pod kątem swing tradingu")
```

`generate_plan()` zwraca słownik z `symbol`/`ticker`, `trading_style`,
`time_horizon`, sekcją `research` oraz `risk_management` (ryzyko 1,5%, SL
1,5 × ATR, minimalne R:R 1:2). `main.py` przekazuje te parametry do
RiskManagera. Metoda `run(AgentContext)` zwraca JSON zgodny z interfejsem MAFSE.
Wyjście modelu jest walidowane przed przekazaniem dalej; plan nie składa
zleceń ani nie stanowi rekomendacji inwestycyjnej.

## Końcowy komunikat

`WriterAgent` z `writer_agent.py` przetwarza dane ResearchAgenta oraz
RiskManagera. W `AgentContext` wyniki należy przekazać pod kluczami `analityk`
i `risk_manager`. Agent waliduje decyzję Gemini i zachowuje poziomy cen oraz
wielkość pozycji wyliczone przez RiskManagera; nie wykonuje zleceń:

```python
from mafse import GeminiClient
from mafse.agent import AgentContext
from writer_agent import WriterAgent

writer = WriterAgent(GeminiClient.from_env())
message = writer.create_trade_message(research_data, risk_data)
```

Wynik zawiera `signal` (KUP/SPRZEDAJ/CZEKAJ), `entry_price`, `stop_loss`,
`take_profit`, `position_size`, `rationale` i czytelny raport `report_text`
po polsku, gotowe do serializacji do JSON. Raport podsumowuje również RSI,
MACD, ATR, interwał oraz szacowane ryzyko; przy danych syntetycznych zawiera
wyraźne ostrzeżenie. Zawiera też `confidence_score_pct` wraz ze składowymi
techniczną, sentymentem potwierdzonych wiadomości i dostępnymi fundamentami,
a także zweryfikowane nagłówki i źródła. Confidence Score jest heurystyczną
oceną zgodności sygnałów, nie skalibrowanym prawdopodobieństwem zysku ani
gwarancją wyniku. Dashboard pokazuje ocenę, jej ograniczenia, fundamenty oraz
linki do wydawców.
