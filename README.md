# Bridge_Strikes-

NYC Bridge/Overpass Truck Strikes Pipeline — Power BI-ready star-schema + ML risk models.

## Project Structure
Bridge_Strikes-/
├── README.md # This file
├── .gitignore # Ignore output/, _pycache_/
├── requirements.txt # pip install -r requirements.txt
├── src/
│ ├── _init_.py
│ ├── pipeline.py # Main script
│ ├── config.py # URLs, constants
│ └── models.py # ML functions
├── tests/
│ ├── _init_.py
│ └── test_pipeline.py
├── docs/
│ └── data_schema.md
├── examples/
│ └── quick_demo.ipynb
└── output/ # Generated CSVs/parquet (gitignored)
