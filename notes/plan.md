
# Magisterka

## tytuł

optymistycznie:
"Improving neural decoding of music from the EEG"
pesymistycznie:
"Neural decoding of music from the EEG with diffusion"

## Plan

Publikacja ["Neural decoding of music from the EEG"](https://www.nature.com/articles/s41598-022-27361-x) dekoduje muzykę, której słucha uczestnik badania, na podstawie mierzonego sygnału EEG.
Zbiór danych (kilka różnych, razem kilkanaście godzin nagrań) z równoczesną muzyką i sygnałem fal mózgowych EEG jest dostępny w internecie.
Autorzy używają sprytnego preprocessingu i sieci bi-LSTM z ~5M parametrów.

Moje pytanie badawcze: Czy da się poprawić wynik używając technik znanych np. z zadań audio generation albo audio source separation?

Konkretnie myślę o architekturze typu: 

 - model A buduje reprezentacje operując na spektrogramie sygnału EEG
 - drugi model B działa jak vocoder który generuje kilkaset milisekund fali dźwiękowej na podstawie odpowiedniego fragmentu outputu modelu A

gdzie model A to może być open-sourcowy model [EEGPT](https://github.com/BINE022/EEGPT) po finetuningu (tj. "foundation" model typu transformer uczony self-supervised, ~10M parametrów).
Z kolei model B to może być model na bazie architektury DiffWave/UNet, który zaczyna od szumu i w procesie dyfuzji generuje fale dźwiękową na podstawie ("conditioned") reprezentacji otrzymanej z modelu A, która powinna kodować spektralne charakterystyki dekodowanej muzyki.

Wątpliwość: brakuje mi doświadczenia by pewnie oszacować czy posiadam wystarczająco danych i mocy obliczeniowej by osiągnąć coś lepszego niż oryginalny LSTM.
Posiadam +100$ do wydania w chmurze dzięki Github Student Pack, a w razie faktycznego otrzymania niezłych wyników i potrzeby na więcej chciałbym się zgłosić o przyznanie czasu komputera w [WCSS](https://wcss.pl/uslugi/25/przetwarzanie-danych-na-superkomputerze/) (czy posiadam do tego prawo jako magistrant?).

### PM

1. Datasets

 - [ ] access nature's dataset
 - [ ] access nmed datasets
 - [ ] access other available datasets (list below)
 - [ ] summarize metadata

2. Pretrained models

 - [ ] access neuro-gpt
 - [ ] access eegpt 1
 - [ ] access eegpt 2
 - [ ] summarize differences



## trening

np. ostatnia warstwa eegpt trenowana.

eeg, music <- batch
A <- eegpt eeg
B <- vocoder A [eeg] ## diffusion loop
loss music B

## plan w punktach

1. Indexable dataset.
 - [ ] combined training and calibration
 - [ ] Random per subject test-val-train split.
 - [ ] common preprocessing
  * [ ] 
 - [ ] Add fmri, scores etc.
2. Preprocessing
3. Basic model: UNet/Diffwave on 0.1-1s chunks training.
4. model: EEGpt finetuned to generate spectrograms
5. Checkpointing. Evalution by ssim.
