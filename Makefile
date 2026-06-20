PYTHON ?= python
TICKER ?= AAPL
START ?= 2020-01-01
END ?= 2024-01-01

.PHONY: install ingest

install:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) src/data_ingest.py --ticker $(TICKER) --start $(START) --end $(END)
