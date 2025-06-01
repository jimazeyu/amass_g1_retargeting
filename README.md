# SMPL Retargeting

This repository provides tools to retarget motion data from the AMASS dataset to G1 humanoid motions.

## 📁 Data Preparation

1. Create a data directory:
   ```bash
   mkdir -p dataset
   ```

2. Download the [AMASS dataset](https://amass.is.tue.mpg.de/):
   - Place the unzipped folders inside `dataset/amass/`.

3. Run the following script to prepare the data:
   ```bash
   ./download_data.sh
   ```

## 🚀 Usage

1. **Process raw AMASS sequences**  
   Collect selected motion sequences from AMASS and save as a single file:
   ```bash
   python process_amass_raw.py
   ```

2. **Split dataset for training/validation/testing**  
   Convert the collected database into train/val/test sets:
   ```bash
   python process_amass_db.py
   ```

3. **Convert to poselib skeleton format**  
   Transform training motion data into poselib joint format:
   ```bash
   python convert_amass_isaac.py
   ```

4. **Retarget to G1 humanoid skeleton**  
   Perform motion retargeting from SMPL-H to G1 humanoid using poselib:
   ```bash
   python retarget_g1_smpl_all.py
   ```

## 🙏 Acknowledgements

- **Retargeting code** is adapted from [ASE](https://github.com/nv-tlabs/ASE).
- **AMASS data processing pipeline** is adapted from [PHC](https://github.com/ZhengyiLuo/PHC).
