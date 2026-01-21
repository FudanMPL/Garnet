#!/bin/bash

# 数据集列表
DATASETS=(
    "kohkiloyeh"
    "diagnosis"
    "iris"
    "wine"
    "cancer"
    "tic-tac-toe"
    "adult"
    "skin-segmentation"
)

# 遍历所有数据集
for DATASET in "${DATASETS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing dataset: $DATASET"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
    
    # ents_xgboost and ents_randomforest (R=32)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Compiling ents_xgboost_efficiency for $DATASET..."
    python ./compile.py -R 32 ents_xgboost_efficiency $DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Compiling ents_randomforest_efficiency for $DATASET..."
    python ./compile.py -R 32 ents_randomforest_efficiency $DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ents_xgboost_efficiency-$DATASET..."
    ./Scripts/ring.sh -F -pn 10000 ents_xgboost_efficiency-$DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ents_randomforest_efficiency-$DATASET..."
    ./Scripts/ring.sh -F -pn 10000 ents_randomforest_efficiency-$DATASET
    
    # hamada_xgboost and hamada_randomforest (R=128)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Compiling hamada_xgboost_efficiency for $DATASET..."
    python ./compile.py -R 128 hamada_xgboost_efficiency $DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Compiling hamada_randomforest_efficiency for $DATASET..."
    python ./compile.py -R 128 hamada_randomforest_efficiency $DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running hamada_xgboost_efficiency-$DATASET..."
    ./Scripts/ring.sh -F -pn 10000 hamada_xgboost_efficiency-$DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running hamada_randomforest_efficiency-$DATASET..."
    ./Scripts/ring.sh -F -pn 10000 hamada_randomforest_efficiency-$DATASET
    
    # abspoel_xgboost and abspoel_randomforest (R=128)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Compiling abspoel_xgboost_efficiency for $DATASET..."
    python ./compile.py -R 128 abspoel_xgboost_efficiency $DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Compiling abspoel_randomforest_efficiency for $DATASET..."
    python ./compile.py -R 128 abspoel_randomforest_efficiency $DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running abspoel_xgboost_efficiency-$DATASET..."
    ./Scripts/ring.sh -F -pn 10000 abspoel_xgboost_efficiency-$DATASET
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running abspoel_randomforest_efficiency-$DATASET..."
    ./Scripts/ring.sh -F -pn 10000 abspoel_randomforest_efficiency-$DATASET
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed processing dataset: $DATASET"
    echo ""
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All datasets processed!"