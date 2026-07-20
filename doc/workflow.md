# Analysis Workflow

## Data Preparation



## Preprocessing

Calculate satellite position and execute single point positioning. Result is saved as a database file.

Observation and navigation RINEX files must be prepared in advance. Coordinate of reference station is also necessary.

```bash
dataset/
├── 0840164k.26g
├── 0840164k.26l
├── 0840164k.26n
├── 0840164k.26o
├── 0840164k.26q
├── 0840164k.nav
├── P115164k.26g
├── P115164k.26l
├── P115164k.26n
├── P115164k.26o
├── P115164k.26q
└── P115164k.nav
```

Run

```bash
$ source .venv/bin/activate
(gnssraw) $ python app/spp.py ./devmemo/data/0840164k.26o ./devmemo/data/0840164k.26n --qzss-nav ./devmemo/data/0840164k.26q --database sample
```

```bash
uv run python app/spp.py ./dataset/P115164k.26o dataset/P115164k.26n --qzss-nav dataset/P115164k.26q --database ./dataset/02P115.db
```

```bash
$ uv run python app/spp.py ./dataset/0840164k.26o dataset/0840164k.26n --qzss-nav dataset/0840164k.26q --database ./dataset/990840.db
```

You can replace satellite position of the database with RTKLIB debug logs. Use RTKLIB CUI tool 'rnx2rtkp' for post processing.

```bash
$ rnx2rtkp -p 0 ./devmemo/data/P115164k.26o ./devmemo/data/P115164k.26n -x 5
```

```bash
$ uv run python misc/update_db_satpos_from_rnx2rtkp_trace.py   --db ./dataset/02P115.db --trace rnx2rtkp.trace   --apply   --diff-csv P115_satpos_diff.csv
```


## Relative positioning


Prepare the configuration file which describes satellite pairs to calculate double-differential observations. Currently, we need trial and error to find a good bias estimation of carrier-phase double differentials. The following calculation takes 2 epochs as input to estimate phase bias.
2 epochs, signal bands, and satellites must be configured in the input JSON file.

```json
{
  "epochs": {
    "start_time": "2026-06-13 10:14:30",
    "end_time": "2026-06-13 10:32:00"
  },
  "bands": [
    "L1"
  ],
  "satellite_blocks": [
    [
      "G10",
      "G15",
      "G20",
      "G24"
    ],
    [
      "J03",
      "J07"
    ]
  ]
}
```

```bash
uv run python app/rp-2epochs.py --base-db dataset/02P115.db --rover-db dataset/990840.db --config-json rp_2epochs_config.json --out out_rp_2epochs.json
```

```bash
uv run python app/rp-bias.py --base-db dataset/990840.db --rover-db dataset/02P115.db --ambiguity-json devmemo/dd_info.json --start 2026-06-13 10:14:30 --end  2026-06-13 10:32:00 --base-pos=-3913066.5486,3483057.8480,3626131.1531
```
