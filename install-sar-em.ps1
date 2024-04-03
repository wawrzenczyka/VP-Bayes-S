pip install -r external/SAR-PU/requirements.txt
pip install -e external/SAR-PU/sarpu
python external/SAR-PU/make_km_lib.py
pip install -e external/SAR-PU/lib/tice
pip install -e external/SAR-PU/lib/km