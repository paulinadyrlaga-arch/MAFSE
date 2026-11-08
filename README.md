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
python3 -m streamlit run app.py
```

Dashboard nie uruchamia skanu automatycznie. Na głównym ekranie można wybrać
„Wszystko”, „Tylko Akcje (Revolut)” lub „Tylko Kryptowaluty (Revolut)”.
Radio pozwala uruchomić tryb na żywo albo „Tryb historyczny (Backtest w wybranym
momencie)”. W trybie historycznym użytkownik wybiera datę, godzinę i minutę w
strefie Europe/Warsaw. ResearchAgent pobiera świece Yahoo Finance z zakresem
kończącym się w tej chwili, odrzuca niedokończone świece źródłowe i oblicza
wskaźniki oraz punktację wyłącznie na danych dostępnych do wybranego momentu.
Yahoo Finance ogranicza historię intraday (typowo ok. 60 dni); starsze daty
mogą nie zwrócić danych. Brak danych jest pokazywany jako błąd, nie mock.
Historyczny skan wyłącza bieżące newsy, fundamenty, arkusze zleceń i kalendarze,
bo ich obecnych wartości nie można traktować jako znanych w przeszłości.
Niedostępność tych filtrów obniża scoring i może blokować kwalifikację.
Na górze ekranu można niezależnie wybrać kategorię horyzontu dla trybu live i
historycznego. „Kilkudniowe” używa świec dziennych z pięcioletniego źródła,
zachowując do 1000 świec do obliczeń, aby SMA50/SMA200 oraz porównanie wolumenu
5/20 sesji miały wystarczającą historię. „Krótkoterminowe / Szybkie” używa
ukończonych świec 15m
(Yahoo Finance z `prepost=True`, jeśli dostawca udostępnia dane przed/po sesji)
w obu trybach. Skan przygotowawczy dla akcji USA rozpoczyna się od 14:00 czasu
polskiego; regularne otwarcie rynku przypada zwykle o 15:30 czasu CET, a w
tygodniach rozbieżności zmiany czasu między Europą i USA o 14:30 czasu
polskiego.
Obie kategorie używają skali 0–150, a sygnał KUP wymaga co najmniej 110 pkt.
Model swing obejmuje: strukturę SMA (25), RSI/MACD (20), wolumen 5/20 sesji
(15), wsparcie (15), konsensus analityków Yahoo Finance (25),
makro/kalendarz/sezonowość/reżim rynku (25) oraz R/R i plan ATR trailing stop
(25). Rosnący MACD porównuje bieżącą wartość z poprzednią świecą.
Model krótkoterminowy obejmuje: RelVol (30), VWAP/POC (25), RSI/histogram
MACD (20), R/R (15), analityków i zweryfikowany news buzz (20), timing/makro
(20) oraz ATR trailing stop/time-stop (20). RelVol jest obliczany dla tej
samej pory z maksymalnie 20 wcześniejszych sesji; poniżej 1,0 zeruje wynik.
Sygnał krótkoterminowy wymaga ponadto R/R co najmniej 1:2.

W trybie live etap techniczny ocenia cały ranking, a dodatkowe źródła sieciowe
są pobierane wyłącznie dla maksymalnie 10 najlepszych kandydatów powyżej
minimalnego wyniku etapu pierwszego. Konsensus pochodzi z bieżących
rekomendacji analityków Yahoo Finance; sentyment newsów uwzględnia tylko
nagłówki potwierdzone przez dwa zaufane źródła. Kalendarz makro korzysta z
publicznego feedu Forex Factory, a terminy raportów spółek z kalendarza Yahoo
Finance. Brak lub niepełny kalendarz nie jest traktowany jako brak ryzyka i
nie przyznaje punktów. W backteście aktualne opinie i kalendarze są pomijane,
aby uniknąć look-ahead.

Pełny ranking jest sortowany malejąco. Rozwijane szczegóły pokazują punktację
każdego filaru, czas świecy, źródła jakościowe oraz wytyczne Revolut. Cel netto
wynosi +4,5% dla swing i +2,0% dla szybkiego modelu. Take-Profit uwzględnia
opłatę kupna i sprzedaży: `entry × (1+fee_in) × (1+target) / (1-fee_out)`.
Stop-Loss początkowy wynosi 1,8×ATR (1.8×ATR). Plan trailingu aktywuje się po
wzroście o 1×ATR i podąża 1,8×ATR za maksimum od wejścia; szybki model dodaje zamknięcie
najpóźniej na koniec sesji. W obu kategoriach kafelek **Potencjalny zysk**
pokazuje szacowaną stopę zwrotu netto od ceny wejścia/Trigger do Take-Profit,
po uwzględnieniu modelowanych opłat za kupno i sprzedaż. Dashboard jedynie
wylicza i wyświetla te poziomy — nie monitoruje otwartej pozycji w tle ani nie
składa zleceń.
Wyniki są heurystyczne, nie są skalibrowanym prawdopodobieństwem ani gwarancją.
Skan używa szerokiej, konfigurowanej w kodzie puli popularnych akcji z USA i
Europy oraz 21 głównych kryptowalut (m.in. BTC, ETH, SOL, XRP, ADA i DOGE),
pobierając OHLCV przez Yahoo Finance na interwale zależnym od kategorii.
Lista tickerów jest
watchlistą Yahoo Finance, nie oficjalnym katalogiem Revolut: dostępność
instrumentów w Revolut zależy od kraju i konta użytkownika.
Rekomendacja w wybranej kategorii wymaga wyniku co najmniej 110/150 i
przejścia dodatkowych warunków kwalifikacyjnych. Karty i szczegółowy widok
pokazują punktację każdego filtra i uzasadnienie. Dane rynkowe
nie są cache’owane: każde uruchomienie skanu pobiera dane ponownie z sieci.
W modelu ustawiono minimum 20 bps round trip, czyli 0,2% (0.2%) łącznie: co najmniej
10 bps (0,1%) prowizji przy zakupie i 10 bps (0,1%) przy sprzedaży. Założenie nie jest pobraną taryfą
Revoluta i nie obejmuje potencjalnego dodatkowego spreadu. Po skanie dashboard pokazuje
pełną listę rankingową wszystkich zeskanowanych aktywów, wraz z tickerami
z błędami pobierania oznaczonymi jako brak wyniku. Ranking jest sortowany
malejąco po punktacji. Wybranie tickera pokazuje cenę w walucie Yahoo Finance,
punktację wybranej kategorii oraz scenariuszowe poziomy Trigger/TP/SL.
Cena wejścia i poziomy transakcji są scenariuszem, nie zleceniem. Yahoo Finance
nie gwarantuje notowań
bez opóźnień; źródło może udostępniać dane opóźnione w zależności od giełdy.
W przypadku problemów z pobraniem danych dashboard pokazuje wyraźny komunikat
błędu i nie zastępuje notowań danymi syntetycznymi.
Brak danych tickerowych jest oznaczany jako błąd i nie jest zastępowany
danymi syntetycznymi. Dane źródłowe mogą być opóźnione lub niedostępne;
wynik skanera nie jest poradą inwestycyjną.

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
Jeśli Yahoo Finance nie zwróci rzeczywistych danych, agent zgłasza błąd zamiast
zwracać dane syntetyczne. Mocki można jawnie włączyć wyłącznie w kodzie
testowym przez `ResearchAgent(allow_mock_data=True)`; wyniki są oznaczone
`is_mock: true` i nie mogą tworzyć rekomendacji.

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
1,8 × ATR, minimalne R:R 1:2). `main.py` przekazuje te parametry do
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
