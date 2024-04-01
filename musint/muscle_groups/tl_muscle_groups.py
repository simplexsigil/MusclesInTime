"""Defines the muscle groups for the thoracolumbar region of the body and the full name of the muscles."""

from itertools import chain

rectus_abdominis_r = {
    "Rectus Abdominis": "TL_rect_abd_r",
}

rectus_abdominis_l = {
    "Rectus Abdominis": "TL_rect_abd_l",
}

external_oblique_r = {
    "External Oblique R5": "TL_E0_R5_r",
    "External Oblique R6": "TL_E0_R6_r",
    "External Oblique R7": "TL_E0_R7_r",
    "External Oblique R8": "TL_E0_R8_r",
    "External Oblique R9": "TL_E0_R9_r",
    "External Oblique R10": "TL_E0_R10_r",
    "External Oblique R11": "TL_E0_R11_r",
    "External Oblique R12": "TL_E0_R12_r",
}

external_oblique_l = {
    "External Oblique R5": "TL_E0_R5_l",
    "External Oblique R6": "TL_E0_R6_l",
    "External Oblique R7": "TL_E0_R7_l",
    "External Oblique R8": "TL_E0_R8_l",
    "External Oblique R9": "TL_E0_R9_l",
    "External Oblique R10": "TL_E0_R10_l",
    "External Oblique R11": "TL_E0_R11_l",
    "External Oblique R12": "TL_E0_R12_l",
}

internal_oblique_r = {
    "Internal Oblique 1": "TL_IO1_r",
    "Internal Oblique 2": "TL_IO2_r",
    "Internal Oblique 3": "TL_IO3_r",
    "Internal Oblique 4": "TL_IO4_r",
    "Internal Oblique 5": "TL_IO5_r",
    "Internal Oblique 6": "TL_IO6_r",
}

internal_oblique_l = {
    "Internal Oblique 1": "TL_IO1_l",
    "Internal Oblique 2": "TL_IO2_l",
    "Internal Oblique 3": "TL_IO3_l",
    "Internal Oblique 4": "TL_IO4_l",
    "Internal Oblique 5": "TL_IO5_l",
    "Internal Oblique 6": "TL_IO6_l",
}

sacro_spinalis_iliocostalis_r = {
    "Sacro Spinalis Iliocostalis 1": "TL_IL_L1_r",
    "Sacro Spinalis Iliocostalis 2": "TL_IL_L2_r",
    "Sacro Spinalis Iliocostalis 3": "TL_IL_L3_r",
    "Sacro Spinalis Iliocostalis 4": "TL_IL_L4_r",
    "Sacro Spinalis Iliocostalis 5": "TL_IL_R5_r",
    "Sacro Spinalis Iliocostalis 6": "TL_IL_R6_r",
    "Sacro Spinalis Iliocostalis 7": "TL_IL_R7_r",
    "Sacro Spinalis Iliocostalis 8": "TL_IL_R8_r",
    "Sacro Spinalis Iliocostalis 9": "TL_IL_R9_r",
    "Sacro Spinalis Iliocostalis 10": "TL_IL_R10_r",
    "Sacro Spinalis Iliocostalis 11": "TL_IL_R11_r",
    "Sacro Spinalis Iliocostalis 12": "TL_IL_R12_r",
}

sacro_spinalis_iliocostalis_l = {
    "Sacro Spinalis Iliocostalis 1": "TL_IL_L1_l",
    "Sacro Spinalis Iliocostalis 2": "TL_IL_L2_l",
    "Sacro Spinalis Iliocostalis 3": "TL_IL_L3_l",
    "Sacro Spinalis Iliocostalis 4": "TL_IL_L4_l",
    "Sacro Spinalis Iliocostalis 5": "TL_IL_R5_l",
    "Sacro Spinalis Iliocostalis 6": "TL_IL_R6_l",
    "Sacro Spinalis Iliocostalis 7": "TL_IL_R7_l",
    "Sacro Spinalis Iliocostalis 8": "TL_IL_R8_l",
    "Sacro Spinalis Iliocostalis 9": "TL_IL_R9_l",
    "Sacro Spinalis Iliocostalis 10": "TL_IL_R10_l",
    "Sacro Spinalis Iliocostalis 11": "TL_IL_R11_l",
    "Sacro Spinalis Iliocostalis 12": "TL_IL_R12_l",
}

sacro_spinalis_longissimus_thoracis_pars_thoracis_r = {
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T1": "TL_LTpT_T1_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T2": "TL_LTpT_T2_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T3": "TL_LTpT_T3_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T4": "TL_LTpT_T4_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T5": "TL_LTpT_T5_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T6": "TL_LTpT_T6_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T7": "TL_LTpT_T7_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T8": "TL_LTpT_T8_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T9": "TL_LTpT_T9_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T10": "TL_LTpT_T10_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T11": "TL_LTpT_T11_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T12": "TL_LTpT_T12_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R4": "TL_LTpT_R4_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R5": "TL_LTpT_R5_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R6": "TL_LTpT_R6_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R7": "TL_LTpT_R7_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R8": "TL_LTpT_R8_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R9": "TL_LTpT_R9_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R10": "TL_LTpT_R10_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R11": "TL_LTpT_R11_r",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R12": "TL_LTpT_R12_r",
}


sacro_spinalis_longissimus_thoracis_pars_thoracis_l = {
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T1": "TL_LTpT_T1_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T2": "TL_LTpT_T2_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T3": "TL_LTpT_T3_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T4": "TL_LTpT_T4_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T5": "TL_LTpT_T5_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T6": "TL_LTpT_T6_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T7": "TL_LTpT_T7_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T8": "TL_LTpT_T8_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T9": "TL_LTpT_T9_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T10": "TL_LTpT_T10_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T11": "TL_LTpT_T11_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis T12": "TL_LTpT_T12_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R4": "TL_LTpT_R4_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R5": "TL_LTpT_R5_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R6": "TL_LTpT_R6_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R7": "TL_LTpT_R7_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R8": "TL_LTpT_R8_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R9": "TL_LTpT_R9_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R10": "TL_LTpT_R10_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R11": "TL_LTpT_R11_l",
    "Sacro Spinalis Longissimus Thoracis pars Thoracis R12": "TL_LTpT_R12_l",
}

sacro_spinalis_longissimus_thoracis_pars_lumbar_r = {
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L5": "TL_LTpL_L5_r",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L4": "TL_LTpL_L4_r",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L3": "TL_LTpL_L3_r",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L2": "TL_LTpL_L2_r",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L1": "TL_LTpL_L1_r",
}

sacro_spinalis_longissimus_thoracis_pars_lumbar_l = {
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L5": "TL_LTpL_L5_l",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L4": "TL_LTpL_L4_l",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L3": "TL_LTpL_L3_l",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L2": "TL_LTpL_L2_l",
    "Sacro Spinalis Longissimus Thoracis pars Lumbar L1": "TL_LTpL_L1_l",
}

multifidus_pelvis_region_r = {
    "Multifidus Pelvis M1T1": "TL_MF_m1t_1_r",
    "Multifidus Pelvis M1T2": "TL_MF_m1t_2_r",
    "Multifidus Pelvis M1t3": "TL_MF_m1t_3_r",
    "Multifidus Pelvis M1S": "TL_MF_m1s_r",
    "Multifidus Pelvis M2S": "TL_MF_m2s_r",
    "Multifidus Pelvis M2T1": "TL_MF_m2t_1_r",
    "Multifidus Pelvis M2T2": "TL_MF_m2t_2_r",
    "Multifidus Pelvis M2T3": "TL_MF_m2t_3_r",
    "Multifidus Pelvis M3S": "TL_MF_m3s_r",
    "Multifidus Pelvis M3t1": "TL_MF_m3t_1_r",
    "Multifidus Pelvis M3t2": "TL_MF_m3t_2_r",
    "Multifidus Pelvis M3t3": "TL_MF_m3t_3_r",
    "Multifidus Pelvis M4S": "TL_MF_m4s_r",
    "Multifidus Pelvis M4T1": "TL_MF_m4t_1_r",
    "Multifidus Pelvis M4T2": "TL_MF_m4t_2_r",
    "Multifidus Pelvis M4T3": "TL_MF_m4t_3_r",
    "Multifidus Pelvis M5S": "TL_MF_m5s_r",
    "Multifidus Pelvis M5T1": "TL_MF_m5t_1_r",
    "Multifidus Pelvis M5T2": "TL_MF_m5t_2_r",
    "Multifidus Pelvis M5T3": "TL_MF_m5t_3_r",
    "Multifidus Pelvis M1 Laminar": "TL_MF_m1_laminar_r",
    "Multifidus Pelvis M2 Laminar": "TL_MF_m2_laminar_r",
    "Multifidus Pelvis M3 Laminar": "TL_MF_m3_laminar_r",
    "Multifidus Pelvis M4 Laminar": "TL_MF_m4_laminar_r",
    "Multifidus Pelvis M5 Laminar": "TL_MF_m5_laminar_r",
}

multifidus_pelvis_region_l = {
    "Multifidus Pelvis M1T1": "TL_MF_m1t_1_l",
    "Multifidus Pelvis M1T2": "TL_MF_m1t_2_l",
    "Multifidus Pelvis M1t3": "TL_MF_m1t_3_l",
    "Multifidus Pelvis M1S": "TL_MF_m1s_l",
    "Multifidus Pelvis M2S": "TL_MF_m2s_l",
    "Multifidus Pelvis M2T1": "TL_MF_m2t_1_l",
    "Multifidus Pelvis M2T2": "TL_MF_m2t_2_l",
    "Multifidus Pelvis M2T3": "TL_MF_m2t_3_l",
    "Multifidus Pelvis M3S": "TL_MF_m3s_l",
    "Multifidus Pelvis M3t1": "TL_MF_m3t_1_l",
    "Multifidus Pelvis M3t2": "TL_MF_m3t_2_l",
    "Multifidus Pelvis M3t3": "TL_MF_m3t_3_l",
    "Multifidus Pelvis M4S": "TL_MF_m4s_l",
    "Multifidus Pelvis M4T1": "TL_MF_m4t_1_l",
    "Multifidus Pelvis M4T2": "TL_MF_m4t_2_l",
    "Multifidus Pelvis M4T3": "TL_MF_m4t_3_l",
    "Multifidus Pelvis M5S": "TL_MF_m5s_l",
    "Multifidus Pelvis M5T1": "TL_MF_m5t_1_l",
    "Multifidus Pelvis M5T2": "TL_MF_m5t_2_l",
    "Multifidus Pelvis M5T3": "TL_MF_m5t_3_l",
    "Multifidus Pelvis M1 Laminar": "TL_MF_m1_laminar_l",
    "Multifidus Pelvis M2 Laminar": "TL_MF_m2_laminar_l",
    "Multifidus Pelvis M3 Laminar": "TL_MF_m3_laminar_l",
    "Multifidus Pelvis M4 Laminar": "TL_MF_m4_laminar_l",
    "Multifidus Pelvis M5 Laminar": "TL_MF_m5_laminar_l",
}


multifidus_neck_region_r = {
    "Multifidus Sup T1-T4": "TL_supmult-T1-C4",
    "Multifidus Sup T1-C5": "TL_supmult-T1-C5",
    "Multifidus Sup T2-C6": "TL_supmult-T2-C6",
    "Multifidus Deep T1-C5": "TL_deepmult-T1-C5",
    "Multifidus Deep T1-C6": "TL_deepmult-T1-C6",
    "Multifidus Deep T2-C7": "TL_deepmult-T2-C7",
    "Multifidus Deep T2-T1": "TL_deepmult-T2-T1",
}

multifidus_neck_region_l = {
    "Multifidus Sup T1-T4": "TL_supmult-T1-C4_L",
    "Multifidus Sup T1-C5": "TL_supmult-T1-C5_L",
    "Multifidus Sup T2-C6": "TL_supmult-T2-C6_L",
    "Multifidus Deep T1-C5": "TL_deepmult-T1-C5_L",
    "Multifidus Deep T1-C6": "TL_deepmult-T1-C6_L",
    "Multifidus Deep T2-C7": "TL_deepmult-T2-C7_L",
    "Multifidus Deep T2-T1": "TL_deepmult-T2-T1_L",
}


multifidus_thoracic_region_r = {
    "Multifidus L4-T12": "TL_multifidus_L4_T12",
    "Multifidus L3-T11": "TL_multifidus_L3_T11",
    "Multifidus T12-T10": "TL_multifidus_T12_T10",
    "Multifidus L2_T10": "TL_multifidus_L2_T10",
    "Multifidus T11_T9": "TL_multifidus_T11_T9",
    "Multifidus L1-T9": "TL_multifidus_L1_T9",
    "Multifidus T10-T8": "TL_multifidus_T10_T8",
    "Multifidus T12-T8": "TL_multifidus_T12_T8",
    "Multifidus T9-T7": "TL_multifidus_T9_T7",
    "Multifidus T11-T7": "TL_multifidus_T11_T7",
    "Multifidus T8-T6": "TL_multifidus_T8_T6",
    "Multifidus T10-T6": "TL_multifidus_T10_T6",
    "Multifidus T7-T5": "TL_multifidus_T7_T5",
    "Multifidus T9-T5": "TL_multifidus_T9_T5",
    "Multifidus T6_T4": "TL_multifidus_T6_T4",
    "Multifidus T8-T4": "TL_multifidus_T8_T4",
    "Multifidus T5-T3": "TL_multifidus_T5_T3",
    "Multifidus T7-T3": "TL_multifidus_T7_T3",
    "Multifidus T4-T2": "TL_multifidus_T4_T2",
    "Multifidus T6-T2": "TL_multifidus_T6_T2",
    "Multifidus T3-T1": "TL_multifidus_T3_T1",
    "Multifidus T5-T1": "TL_multifidus_T5_T1",
    "Multifidus T4-C7": "TL_multifidus_T4_C7",
}

multifidus_thoracic_region_l = {
    "Multifidus L4-T12": "TL_multifidus_L4_T12_L",
    "Multifidus L3-T11": "TL_multifidus_L3_T11_L",
    "Multifidus T12-T10": "TL_multifidus_T12_T10_L",
    "Multifidus L2_T10": "TL_multifidus_L2_T10_L",
    "Multifidus T11_T9": "TL_multifidus_T11_T9_L",
    "Multifidus L1-T9": "TL_multifidus_L1_T9_L",
    "Multifidus T10-T8": "TL_multifidus_T10_T8_L",
    "Multifidus T12-T8": "TL_multifidus_T12_T8_L",
    "Multifidus T9-T7": "TL_multifidus_T9_T7_L",
    "Multifidus T11-T7": "TL_multifidus_T11_T7_L",
    "Multifidus T8-T6": "TL_multifidus_T8_T6_L",
    "Multifidus T10-T6": "TL_multifidus_T10_T6_L",
    "Multifidus T7-T5": "TL_multifidus_T7_T5_L",
    "Multifidus T9-T5": "TL_multifidus_T9_T5_L",
    "Multifidus T6_T4": "TL_multifidus_T6_T4_L",
    "Multifidus T8-T4": "TL_multifidus_T8_T4_L",
    "Multifidus T5-T3": "TL_multifidus_T5_T3_L",
    "Multifidus T7-T3": "TL_multifidus_T7_T3_L",
    "Multifidus T4-T2": "TL_multifidus_T4_T2_L",
    "Multifidus T6-T2": "TL_multifidus_T6_T2_L",
    "Multifidus T3-T1": "TL_multifidus_T3_T1_L",
    "Multifidus T5-T1": "TL_multifidus_T5_T1_L",
    "Multifidus T4-C7": "TL_multifidus_T4_C7_L",
}

psoas_r = {
    "Psoas L1 VB": "TL_Ps_L1_VB_r",
    "Psoas L1 TP": "TL_Ps_L1_TP_r",
    "Psoas L1 L2 IVD": "TL_Ps_L1_L2_IVD_r",
    "Psoas L2 TP": "TL_Ps_L2_TP_r",
    "Psoas L2 L3 IVD": "TL_Ps_L2_L3_IVD_r",
    "Psoas L3 TP": "TL_Ps_L3_TP_r",
    "Psoas L3 L4 IVD": "TL_Ps_L3_L4_IVD_r",
    "Psoas L4 TP": "TL_Ps_L4_TP_r",
    "Psoas L4 L5 IVD": "TL_Ps_L4_L5_IVD_r",
    "Psoas L5 TP": "TL_Ps_L5_TP_r",
    "Psoas L5 VB": "TL_Ps_L5_VB_r",
}

psoas_l = {
    "Psoas L1 VB": "TL_Ps_L1_VB_l",
    "Psoas L1 TP": "TL_Ps_L1_TP_l",
    "Psoas L1 L2 IVD": "TL_Ps_L1_L2_IVD_l",
    "Psoas L2 TP": "TL_Ps_L2_TP_l",
    "Psoas L2 L3 IVD": "TL_Ps_L2_L3_IVD_l",
    "Psoas L3 TP": "TL_Ps_L3_TP_l",
    "Psoas L3 L4 IVD": "TL_Ps_L3_L4_IVD_l",
    "Psoas L4 TP": "TL_Ps_L4_TP_l",
    "Psoas L4 L5 IVD": "TL_Ps_L4_L5_IVD_l",
    "Psoas L5 TP": "TL_Ps_L5_TP_l",
    "Psoas L5 VB": "TL_Ps_L5_VB_l",
}

quadratus_lumborum_r = {
    "Quadratus Lumborum Post I 1-L3": "TL_QL_post_I_1-L3_r",
    "Quadratus Lumborum Post I 2-L4": "TL_QL_post_I_2-L4_r",
    "Quadratus Lumborum Post I 2-L3": "TL_QL_post_I_2-L3_r",
    "Quadratus Lumborum Post I 2-L2": "TL_QL_post_I_2-L2_r",
    "Quadratus Lumborum Post I 3-L1": "TL_QL_post_I_3-L1_r",
    "Quadratus Lumborum Post I 3-L2": "TL_QL_post_I_3-L2_r",
    "Quadratus Lumborum Post I 3-L3": "TL_QL_post_I_3-L3_r",
    "Quadratus Lumborum Mid L3-12 3": "TL_QL_mid_L3-12_3_r",
    "Quadratus Lumborum Mid L3-12 2": "TL_QL_mid_L3-12_2_r",
    "Quadratus Lumborum Mid L3-12 1": "TL_QL_mid_L3-12_1_r",
    "Quadratus Lumborum Mid L2-12 1": "TL_QL_mid_L2-12_1_r",
    "Quadratus Lumborum Mid L2-12 3": "TL_QL_mid_L4-12_3_r",
    "Quadratus Lomborum Anterior I 2-T12": "TL_QL_ant_I_2-T12_r",
    "Quadratus Lomborum Anterior I 3-T12": "TL_QL_ant_I_3-T12_r",
    "Quadratus Lomborum Anterior I 2-12 1": "TL_QL_ant_I_2-12_1_r",
    "Quadratus Lomborum Anterior I 3-12 1": "TL_QL_ant_I_3-12_1_r",
    "Quadratus Lomborum Anterior I 3-12 2": "TL_QL_ant_I_3-12_2_r",
    "Quadratus Lomborum Anterior I 3-12 3": "TL_QL_ant_I_3-12_3_r",
}

quadratus_lumborum_l = {
    "Quadratus Lumborum Post I 1-L3": "TL_QL_post_I_1-L3_l",
    "Quadratus Lumborum Post I 2-L4": "TL_QL_post_I_2-L4_l",
    "Quadratus Lumborum Post I 2-L3": "TL_QL_post_I_2-L3_l",
    "Quadratus Lumborum Post I 2-L2": "TL_QL_post_I_2-L2_l",
    "Quadratus Lumborum Post I 3-L1": "TL_QL_post_I_3-L1_l",
    "Quadratus Lumborum Post I 3-L2": "TL_QL_post_I_3-L2_l",
    "Quadratus Lumborum Post I 3-L3": "TL_QL_post_I_3-L3_l",
    "Quadratus Lumborum Mid L3-12 3": "TL_QL_mid_L3-12_3_l",
    "Quadratus Lumborum Mid L3-12 2": "TL_QL_mid_L3-12_2_l",
    "Quadratus Lumborum Mid L3-12 1": "TL_QL_mid_L3-12_1_l",
    "Quadratus Lumborum Mid L2-12 1": "TL_QL_mid_L2-12_1_l",
    "Quadratus Lumborum Mid L2-12 3": "TL_QL_mid_L4-12_3_l",
    "Quadratus Lomborum Anterior I 2-T12": "TL_QL_ant_I_2-T12_l",
    "Quadratus Lomborum Anterior I 3-T12": "TL_QL_ant_I_3-T12_l",
    "Quadratus Lomborum Anterior I 2-12 1": "TL_QL_ant_I_2-12_1_l",
    "Quadratus Lomborum Anterior I 3-12 1": "TL_QL_ant_I_3-12_1_l",
    "Quadratus Lomborum Anterior I 3-12 2": "TL_QL_ant_I_3-12_2_l",
    "Quadratus Lomborum Anterior I 3-12 3": "TL_QL_ant_I_3-12_3_l",
}

sternocleidomastoideus_r = {
    "Sternocleidomastoideus Mastoid": "TL_stern_mast",
    "Sternocleidomastoideus Clavicle": "TL_cleid_mast",
    "Sternocleidomastoideus Occipital": "TL_cleid_occ",
}

sternocleidomastoideus_l = {
    "Sternocleidomastoideus Mastoid": "TL_stern_mast_L",
    "Sternocleidomastoideus Clavicle": "TL_cleid_mast_L",
    "Sternocleidomastoideus Occipital": "TL_cleid_occ_L",
}

scalenus_r = {
    "Scalenus Anterior": "TL_scalenus_ant",
    "Scalenus Medial": "TL_scalenus_med",
    "Scalenus Posterior": "TL_scalenus_post",
}

scalenus_l = {
    "Scalenus Anterior": "TL_scalenus_ant_L",
    "Scalenus Medial": "TL_scalenus_med_L",
    "Scalenus Posterior": "TL_scalenus_post_L",
}

longus_colli_r = {
    "Longus Colli C1-Thx": "TL_long_col_c1thx",
    "Longus Colli C5-Thx": "TL_long_col_c5thx",
}

longus_colli_l = {
    "Longus Colli C1-Thx": "TL_long_col_c1thx_L",
    "Longus Colli C5-Thx": "TL_long_col_c5thx_L",
}

splenius_capitis_r = {
    "Splenius Capitis T1": "TL_splen_cap_skl_T1",
    "Splenius Capitis T2": "TL_splen_cap_skl_T2",
}

splenius_capitis_l = {
    "Splenius Capitis T1": "TL_splen_cap_skl_T1_L",
    "Splenius Capitis T2": "TL_splen_cap_skl_T2_L",
}

splenius_cervicis_r = {
    "Splenius Cervicis C3 T3": "TL_splen_cerv_c3_T3",
    "Splenius Cervicis C3 T4": "TL_splen_cerv_c3_T4",
    "Splenius Cervicis C3 T5": "TL_splen_cerv_c3_T5",
    "Splenius Cervicis C3 T6": "TL_splen_cerv_c3_T6",
}

splenius_cervicis_l = {
    "Splenius Cervicis C3 T3": "TL_splen_cerv_c3_T3_L",
    "Splenius Cervicis C3 T4": "TL_splen_cerv_c3_T4_L",
    "Splenius Cervicis C3 T5": "TL_splen_cerv_c3_T5_L",
    "Splenius Cervicis C3 T6": "TL_splen_cerv_c3_T6_L",
}

semispinalis_capitis_r = {
    "Semispinalis Capitis Skull Thorax": "TL_semi_cap_sklthx",
}

semispinalis_capitis_l = {
    "Semispinalis Capitis Skull Thorax": "TL_semi_cap_sklthx_L",
}

semispinalis_cervicis_r = {
    "Semispinalis Cervicis C3 Thorax": "TL_semi_cerv_c3thx",
}

semispinalis_cervicis_l = {
    "Semispinalis Cervicis C3 Thorax": "TL_semi_cerv_c3thx_L",
}

levator_scapulae_r = {
    "Levator Scapulae": "TL_levator_scap",
}

levator_scapulae_l = {
    "Levator Scapulae": "TL_levator_scap_L",
}

longissi_cervicis_r = {
    "Longissi Cervicis C4 Thorax": "TL_longissi_cerv_c4thx",
}

longissi_cervicis_l = {
    "Longissi Cervicis C4 Thorax": "TL_longissi_cerv_c4thx_L",
}

iliocostalis_cervicis_r = {
    "Iliocostalis Cervicis C5 Rib": "TL_iliocost_cerv_c5rib",
}

iliocostalis_cervicis_l = {
    "Iliocostalis Cervicis C5 Rib": "TL_iliocost_cerv_c5rib_L",
}

transversus_abdominus_r = {
    "Transversus 1": "TL_TR1_r",
    "Transversus 2": "TL_TR2_r",
    "Transversus 3": "TL_TR3_r",
    "Transversus 4": "TL_TR4_r",
    "Transversus 5": "TL_TR5_r",
}


transversus_abdominus_l = {
    "Transversus 1": "TL_TR1_l",
    "Transversus 2": "TL_TR2_l",
    "Transversus 3": "TL_TR3_l",
    "Transversus 4": "TL_TR4_l",
    "Transversus 5": "TL_TR5_l",
}


core_and_abdominal_r_muscles = {
    "Rectus Abdominis": list(rectus_abdominis_r.values()),
    "External Oblique": list(external_oblique_r.values()),
    "Internal Oblique": list(internal_oblique_r.values()),
    "Transversus Abdominus": list(transversus_abdominus_r.values()),
}

core_and_abdominal_l_muscles = {
    "Rectus Abdominis": list(rectus_abdominis_l.values()),
    "External Oblique": list(external_oblique_l.values()),
    "Internal Oblique": list(internal_oblique_l.values()),
    "Transversus Abdominus": list(transversus_abdominus_l.values()),
}

back_and_spinal_r_muscles = {
    "Sacro Spinalis Iliocostalis": list(sacro_spinalis_iliocostalis_r.values()),
    "Sacro Spinalis Longissimus Thoracis Pars Thoracis": list(
        sacro_spinalis_longissimus_thoracis_pars_thoracis_r.values()
    ),
    "Sacro Spinalis Longissimus Thoracis Pars Lumbar": list(
        sacro_spinalis_longissimus_thoracis_pars_lumbar_r.values()
    ),
    "Multifidus Pelvis Region": list(multifidus_pelvis_region_r.values()),
    "Multifidus Neck Region": list(multifidus_neck_region_r.values()),
    "Multifidus Thoracic Region": list(multifidus_thoracic_region_r.values()),
}

back_and_spinal_l_muscles = {
    "Sacro Spinalis Iliocostalis": list(sacro_spinalis_iliocostalis_l.values()),
    "Sacro Spinalis Longissimus Thoracis Pars Thoracis": list(
        sacro_spinalis_longissimus_thoracis_pars_thoracis_l.values()
    ),
    "Sacro Spinalis Longissimus Thoracis Pars Lumbar": list(
        sacro_spinalis_longissimus_thoracis_pars_lumbar_l.values()
    ),
    "Multifidus Pelvis Region": list(multifidus_pelvis_region_l.values()),
    "Multifidus Neck Region": list(multifidus_neck_region_l.values()),
    "Multifidus Thoracic Region": list(multifidus_thoracic_region_l.values()),
}

hip_and_pelvic_r_muscles = {
    "Psoas": list(psoas_r.values()),
    "Quadratus Lumborum Posterior": list(quadratus_lumborum_r.values()),
}

hip_and_pelvic_l_muscles = {
    "Psoas": list(psoas_l.values()),
    "Quadratus Lumborum Posterior": list(quadratus_lumborum_l.values()),
}

neck_l_muscles = {
    "Sternocleidomastoid": list(sternocleidomastoideus_r.values()),
    "Scalenus": list(scalenus_r.values()),
    "Longus Colli": list(longus_colli_r.values()),
    "Splenius Capitis": list(splenius_capitis_r.values()),
    "Splenius Cervicis": list(splenius_cervicis_r.values()),
    "Levator Scapulae": list(levator_scapulae_r.values()),
    "Semispinalis Capitis": list(semispinalis_capitis_r.values()),
}

neck_r_muscles = {
    "Sternocleidomastoid": list(sternocleidomastoideus_l.values()),
    "Scalenus": list(scalenus_l.values()),
    "Longus Colli": list(longus_colli_l.values()),
    "Splenius Capitis": list(splenius_capitis_l.values()),
    "Splenius Cervicis": list(splenius_cervicis_l.values()),
    "Levator Scapulae": list(levator_scapulae_l.values()),
    "Semispinalis Capitis": list(semispinalis_capitis_l.values()),
}

thoracic_and_cervical_r_muscles = {
    "Longissimus Cervicis": list(longissi_cervicis_r.values()),
    "Iliocostalis Cervicis": list(iliocostalis_cervicis_r.values()),
    "Semispinalis Cervicis": list(semispinalis_cervicis_r.values()),
    "Splenius Cervicis": list(splenius_cervicis_r.values()),
}

thoracic_and_cervical_l_muscles = {
    "Longissimus Cervicis": list(longissi_cervicis_l.values()),
    "Iliocostalis Cervicis": list(iliocostalis_cervicis_l.values()),
    "Semispinalis Cervicis": list(semispinalis_cervicis_l.values()),
    "Splenius Cervicis": list(splenius_cervicis_l.values()),
}


tl_muscle_groups_level1 = {
    "right core": list(chain.from_iterable(core_and_abdominal_r_muscles.values())),
    "right lower back": list(chain.from_iterable(back_and_spinal_r_muscles.values())),
    "right hip": list(chain.from_iterable(hip_and_pelvic_r_muscles.values())),
    "right neck": list(chain.from_iterable(neck_r_muscles.values())),
    "right upper back": list(
        chain.from_iterable(thoracic_and_cervical_r_muscles.values())
    ),
    "left core": list(chain.from_iterable(core_and_abdominal_l_muscles.values())),
    "left lower back": list(chain.from_iterable(back_and_spinal_l_muscles.values())),
    "left hip": list(chain.from_iterable(hip_and_pelvic_l_muscles.values())),
    "left neck": list(chain.from_iterable(neck_l_muscles.values())),
    "left upper back": list(
        chain.from_iterable(thoracic_and_cervical_l_muscles.values())
    ),
}


class ThoracolumbarMuscleGroups:
    def __init__(self):
        self.level1 = tl_muscle_groups_level1

    def get_muscle_group(self, level, group):
        if level == 1:
            return self.level1[group]
        else:
            return None

    def get_muscle_groups(self, level):
        if level == 1:
            return self.level1
        else:
            return None
        

if __name__ == "__main__":
    tl_muscle_groups = ThoracolumbarMuscleGroups()
    print(tl_muscle_groups.get_muscle_group(1, "right core and abdominal muscles"))
    print(tl_muscle_groups.get_muscle_groups(1))