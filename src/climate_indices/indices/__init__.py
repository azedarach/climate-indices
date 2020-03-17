"""
Provides routines for computing climate indices.
"""


from __future__ import absolute_import


from .ao import ao_loading_pattern, pc_ao
from .enso import (calculate_mei_standardized_seasonal_anomaly,
                   mei, mei_loading_pattern, nino1, nino2, nino3,
                   nino34, nino4, restrict_to_mei_analysis_region,
                   soi, troup_soi)
from .sam import gong_wang_sam, marshall_sam, pc_sam, sam_loading_pattern
from .indopacific_sst import dc_sst, dc_sst_loading_pattern
from .iod import dmi, zwi
from .mjo import wh_rmm, wh_rmm_anomalies, wh_rmm_eofs
from .nao import hurrell_nao
from .nhtele import (calculate_kmeans_pcs_anomalies,
                     kmeans_pc_clustering, kmeans_pcs,
                     kmeans_pcs_composites)
from .pna import (modified_pointwise_pna4, pc_pna, pointwise_pna3,
                  pointwise_pna4)
from .psa import real_pc_psa1
