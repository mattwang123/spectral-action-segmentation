# spectral-action-segmentation
This is the final project for 110.445 MCFDS.

### Environment Setup
```bash
conda create -n mfds python=3.8
conda activate mfds
pip install -r requirements.txt
```

### example usage:
change video path
```bash
python preprocess_data.py
python baseline\kmeans.py
```