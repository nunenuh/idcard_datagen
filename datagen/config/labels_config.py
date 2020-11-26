LABELS = [
    'U-FLD_PROV', 'B-VAL_PROV', 'L-VAL_PROV', 'U-FLD_KAB', 'U-VAL_KAB',
    'U-FLD_NIK', 'U-VAL_NIK', 'U-FLD_NAMA', 'B-VAL_NAMA', 'L-VAL_NAMA',
    'B-FLD_TTL', 'L-FLD_TTL', 'B-VAL_TTL', 'L-VAL_TTL', 'B-FLD_GDR',
    'L-FLD_GDR', 'U-VAL_GDR', 'B-FLD_GLD', 'L-FLD_GLD', 'U-VAL_GLD',
    'U-FLD_ADR', 'B-VAL_ADR', 'I-VAL_ADR', 'L-VAL_ADR', 'U-FLD_RTW',
    'U-VAL_RTW', 'U-FLD_KLH', 'U-VAL_KLH', 'U-FLD_KCM', 'U-VAL_KCM',
    'U-FLD_RLG', 'U-VAL_RLG', 'B-FLD_KWN', 'L-FLD_KWN', 'B-VAL_KWN',
    'L-VAL_KWN', 'U-FLD_KRJ', 'U-VAL_KRJ', 'U-FLD_WRG', 'U-VAL_WRG',
    'B-FLD_BLK', 'L-FLD_BLK', 'B-VAL_BLK', 'L-VAL_BLK', 'U-VAL_SGP',
    'U-VAL_SGD', 'B-VAL_KAB', 'L-VAL_KAB', 'U-VAL_NAMA', 'B-VAL_KLH',
    'L-VAL_KLH', 'B-VAL_KRJ', 'I-VAL_KRJ', 'L-VAL_KRJ', 'B-VAL_SGP',
    'L-VAL_SGP', 'I-VAL_TTL', 'L-VAL_KCM', 'B-VAL_KCM', 'U-VAL_KWN',
    'U-VAL_PROV', 'I-VAL_NAMA', 'I-VAL_PROV', 'I-VAL_KAB', 'I-VAL_KCM',
    'I-VAL_SGP', 'U-VAL_ADR', 'I-VAL_KLH', 'O'
]

LABEL2INDEX = dict((label,idx) for idx, label in enumerate(LABELS))
INDEX2LABEL = dict((idx, label) for idx, label in enumerate(LABELS))
NUM_LABELS = len(LABELS)