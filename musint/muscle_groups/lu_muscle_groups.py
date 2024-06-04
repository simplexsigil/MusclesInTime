"""Defines muscle groups for the lower body and the full names of the muscles"""

from itertools import chain

gluteus_maximus_r = {
    "Gluteus Maximus 1": "LU_glmax1_r",
    "Gluteus Maximus 2": "LU_glmax2_r",
    "Gluteus Maximus 3": "LU_glmax3_r",
}

gluteus_medius_r = {
    "Gluteus Medius 1": "LU_glmed1_r",
    "Gluteus Medius 2": "LU_glmed2_r",
    "Gluteus Medius 3": "LU_glmed3_r",
}

gluteus_minimus_r = {
    "Gluteus Minimus 1": "LU_glmin1_r",
    "Gluteus Minimus 2": "LU_glmin2_r",
    "Gluteus Minimus 3": "LU_glmin3_r",
}

adductor_brevis_r = {
    "Adductor Brevis": "LU_addbrev_r",
}

adductor_longus_r = {
    "Adductor Longus": "LU_addlong_r",
}

adductor_magnus_distal_r = {
    "Adductor Magnus Distal": "LU_addmagDist_r",
}

adductor_magnus_ischial_r = {
    "Adductor Magnus Ischial": "LU_addmagIsch_r",
}

adductor_magnus_mid_r = {
    "Adductor Magnus Mid": "LU_addmagMid_r",
}

adductor_magnus_proximal_r = {
    "Adductor Magnus Proximal": "LU_addmagProx_r",
}

biceps_femoris_long_head_r = {
    "Biceps Femoris Long Head": "LU_bflh_r",
}


biceps_fermoris_short_head_r = {
    "Biceps Femoris Short Head": "LU_bfsh_r",
}

gracilis_r = {
    "Gracilis": "LU_grac_r",
}

semitendinosus_r = {
    "Semitendinosus": "LU_semiten_r",
}

semimembranosus_r = {
    "Semimembranosus": "LU_semimem_r",
}

tensor_fasciae_latae_r = {
    "Tensor Fasciae Latae": "LU_tfl_r",
}

piriformis_r = {
    "Piriformis": "LU_piri_r",
}

sartorius_r = {
    "Sartorius": "LU_sart_r",
}

iliacus_r = {
    "Iliacus": "LU_iliacus_r",
}

psoas_r = {
    "Psoas": "LU_psoas_r",
}

recfem_r = {
    "Rectus Femoris": "LU_recfem_r",
}

gastrocnemius_lateral_r = {
    "Gastrocnemius Lateral": "LU_gaslat_r",
}

gastrocnemius_medial_r = {
    "Gastrocnemius Medial": "LU_gasmed_r",
}

vastus_intermedius_r = {
    "Vastus Intermedius": "LU_vasint_r",
}

vastus_lateralis_r = {
    "Vastus Lateralis": "LU_vaslat_r",
}

vastus_medialis_r = {
    "Vastus Medialis": "LU_vasmed_r",
}

extensor_digitorum_longus_r = {
    "Extensor Digitorum Longus": "LU_edl_r",
}

extensor_hallucis_longus_r = {
    "Extensor Hallucis Longus": "LU_ehl_r",
}

tibialis_anterior_r = {
    "Tibialis Anterior": "LU_tibant_r",
}

flexor_digitorum_longus_r = {
    "Flexor Digitorum Longus": "LU_fdl_r",
}

flexor_hallucis_longus_r = {
    "Flexor Hallucis Longus": "LU_fhl_r",
}

per_brev_r = {
    "Peroneus Brevis": "LU_perbrev_r",
}

per_long_r = {
    "Peroneus Longus": "LU_perlong_r",
}

soleus_r = {
    "Soleus": "LU_soleus_r",
}

gluteus_maximus_l = {
    "Gluteus Maximus 1": "LU_glmax1_l",
    "Gluteus Maximus 2": "LU_glmax2_l",
    "Gluteus Maximus 3": "LU_glmax3_l",
}

gluteus_medius_l = {
    "Gluteus Medius 1": "LU_glmed1_l",
    "Gluteus Medius 2": "LU_glmed2_l",
    "Gluteus Medius 3": "LU_glmed3_l",
}

gluteus_minimus_l = {
    "Gluteus Minimus 1": "LU_glmin1_l",
    "Gluteus Minimus 2": "LU_glmin2_l",
    "Gluteus Minimus 3": "LU_glmin3_l",
}

adductor_brevis_l = {
    "Adductor Brevis": "LU_addbrev_l",
}

adductor_longus_l = {
    "Adductor Longus": "LU_addlong_l",
}

adductor_magnus_distal_l = {
    "Adductor Magnus Distal": "LU_addmagDist_l",
}

adductor_magnus_ischial_l = {
    "Adductor Magnus Ischial": "LU_addmagIsch_l",
}

adductor_magnus_mid_l = {
    "Adductor Magnus Mid": "LU_addmagMid_l",
}

adductor_magnus_proximal_l = {
    "Adductor Magnus Proximal": "LU_addmagProx_l",
}

biceps_femoris_long_head_l = {
    "Biceps Femoris Long Head": "LU_bflh_l",
}

biceps_fermoris_short_head_l = {
    "Biceps Femoris Short Head": "LU_bfsh_l",
}

gracilis_l = {
    "Gracilis": "LU_grac_l",
}

semitendinosus_l = {
    "Semitendinosus": "LU_semiten_l",
}

semimembranosus_l = {
    "Semimembranosus": "LU_semimem_l",
}

tensor_fasciae_latae_l = {
    "Tensor Fasciae Latae": "LU_tfl_l",
}

piriformis_l = {
    "Piriformis": "LU_piri_l",
}

sartorius_l = {
    "Sartorius": "LU_sart_l",
}

iliacus_l = {
    "Iliacus": "LU_iliacus_l",
}

psoas_l = {
    "Psoas": "LU_psoas_l",
}

recfem_l = {
    "Rectus Femoris": "LU_recfem_l",
}

gastrocnemius_lateral_l = {
    "Gastrocnemius Lateral": "LU_gaslat_l",
}

gastrocnemius_medial_l = {
    "Gastrocnemius Medial": "LU_gasmed_l",
}

vastus_intermedius_l = {
    "Vastus Intermedius": "LU_vasint_l",
}

vastus_lateralis_l = {
    "Vastus Lateralis": "LU_vaslat_l",
}

vastus_medialis_l = {
    "Vastus Medialis": "LU_vasmed_l",
}

extensor_digitorum_longus_l = {
    "Extensor Digitorum Longus": "LU_edl_l",
}

extensor_hallucis_longus_l = {
    "Extensor Hallucis Longus": "LU_ehl_l",
}

tibialis_anterior_l = {
    "Tibialis Anterior": "LU_tibant_l",
}

flexor_digitorum_longus_l = {
    "Flexor Digitorum Longus": "LU_fdl_l",
}

flexor_hallucis_longus_l = {
    "Flexor Hallucis Longus": "LU_fhl_l",
}

per_brev_l = {
    "Peroneus Brevis": "LU_perbrev_l",
}

per_long_l = {
    "Peroneus Longus": "LU_perlong_l",
}

soleus_l = {
    "Soleus": "LU_soleus_l",
}

hip_adduction_r_muscles = {
    "Adductor Brevis": list(adductor_brevis_r.values()),
    "Adductor Longus": list(adductor_longus_r.values()),
    "Adductor Magnus Distal": list(adductor_magnus_distal_r.values()),
    "Adductor Magnus Ischial": list(adductor_magnus_ischial_r.values()),
    "Adductor Magnus Mid": list(adductor_magnus_mid_r.values()),
    "Adductor Magnus Proximal": list(adductor_magnus_proximal_r.values()),
    "Biceps Femoris Long Head": list(biceps_femoris_long_head_r.values()),
    "Gracilis": list(gracilis_r.values()),
    "Semitendinosus": list(semitendinosus_r.values()),
    "Semimembranosus": list(semimembranosus_r.values()),
}

hip_abduction_r_muscles = {
    "Gluteus Maximus": list(gluteus_maximus_r.values()),
    "Gluteus Medius": list(gluteus_medius_r.values()),
    "Gluteus Minimus": list(gluteus_minimus_r.values()),
    "Tensor Fasciae Latae": list(tensor_fasciae_latae_r.values()),
    "Piriformis": list(piriformis_r.values()),
    "Sartorius": list(sartorius_r.values()),
}

hip_flexion_r_muscles = {
    "Adductor Brevis": list(adductor_brevis_r.values()),
    "Adductor Longus": list(adductor_longus_r.values()),
    "Gluteus Minimus": list(gluteus_minimus_r.values()),
    "Gracilis": list(gracilis_r.values()),
    "Iliacus": list(iliacus_r.values()),
    "Psoas": list(psoas_r.values()),
    "Rectus Femoris": list(recfem_r.values()),
    "Sartorius": list(sartorius_r.values()),
    "Tensor Fasciae Latae": list(tensor_fasciae_latae_r.values()),
}

hip_extension_r_muscles = {
    "Adductor Longus": list(adductor_longus_r.values()),
    "Adductor Magnus Distal": list(adductor_magnus_distal_r.values()),
    "Adductor Magnus Ischial": list(adductor_magnus_ischial_r.values()),
    "Adductor Magnus Mid": list(adductor_magnus_mid_r.values()),
    "Adductor Magnus Proximal": list(adductor_magnus_proximal_r.values()),
    "Biceps Femoris Long Head": list(biceps_femoris_long_head_r.values()),
    "Gluteus Maximus": list(gluteus_maximus_r.values()),
    "Gluteus Medius": list(gluteus_medius_r.values()),
    "Gluteus Minimus": list(gluteus_minimus_r.values()),
    "Semitendinosus": list(semitendinosus_r.values()),
    "Semimembranosus": list(semimembranosus_r.values()),
}

hip_inrotation_r_muscles = {
    "Gluteus Medius": list(gluteus_medius_r.values()),
    "Gluteus Minimus": list(gluteus_minimus_r.values()),
    "Iliacus": list(iliacus_r.values()),
    "Psoas": list(psoas_r.values()),
    "Tensor Fasciae Latae": list(tensor_fasciae_latae_r.values()),
}

hip_exrotation_r_muscles = {
    "Gluteus Minimus": list(gluteus_minimus_r.values()),
    "Piriformis": list(piriformis_r.values()),
}

knee_flexion_r_muscles = {
    "Biceps Femoris Long Head": list(biceps_femoris_long_head_r.values()),
    "Biceps Femoris Short Head": list(biceps_fermoris_short_head_r.values()),
    "Gastrocnemius Lateral": list(gastrocnemius_lateral_r.values()),
    "Gastrocnemius Medial": list(gastrocnemius_medial_r.values()),
    "Gracilis": list(gracilis_r.values()),
    "Sartorius": list(sartorius_r.values()),
    "Semitendinosus": list(semitendinosus_r.values()),
    "Semimembranosus": list(semimembranosus_r.values()),
}

knee_extension_r_muscles = {
    "Rectus Femoris": list(recfem_r.values()),
    "Vastus Intermedius": list(vastus_intermedius_r.values()),
    "Vastus Lateralis": list(vastus_lateralis_r.values()),
    "Vastus Medialis": list(vastus_medialis_r.values()),
}

ankle_dorsiflexion_r_muscles = {
    "Extensor Digitorum Longus": list(extensor_digitorum_longus_r.values()),
    "Extensor Hallucis Longus": list(extensor_hallucis_longus_r.values()),
    "Tibialis Anterior": list(tibialis_anterior_r.values()),
}

ankle_plantarflexion_r_muscles = {
    "Flexor Digitorum Longus": list(flexor_digitorum_longus_r.values()),
    "Flexor Hallucis Longus": list(flexor_hallucis_longus_r.values()),
    "Gastrocnemius Lateral": list(gastrocnemius_lateral_r.values()),
    "Gastrocnemius Medial": list(gastrocnemius_medial_r.values()),
    "Peroneus Brevis": list(per_brev_r.values()),
    "Peroneus Longus": list(per_long_r.values()),
    "Soleus": list(soleus_r.values()),
    "Tibialis Posterior": list(tibialis_anterior_r.values()),
}

evertor_r_muscles = {
    "Extensor Digitorum Longus": list(extensor_digitorum_longus_r.values()),
    "Peroneus Brevis": list(per_brev_r.values()),
    "Peroneus Longus": list(per_long_r.values()),
}

invertor_r_muscles = {
    "Extensor Hallucis Longus": list(extensor_hallucis_longus_r.values()),
    "Flexor Digitorum Longus": list(flexor_digitorum_longus_r.values()),
    "Flexor Hallucis Longus": list(flexor_hallucis_longus_r.values()),
    "Tibialis Anterior": list(tibialis_anterior_r.values()),
    "Tibialis Posterior": list(tibialis_anterior_r.values()),
}

hip_adduction_l_muscles = {
    "Adductor Brevis": list(adductor_brevis_l.values()),
    "Adductor Longus": list(adductor_longus_l.values()),
    "Adductor Magnus Distal": list(adductor_magnus_distal_l.values()),
    "Adductor Magnus Ischial": list(adductor_magnus_ischial_l.values()),
    "Adductor Magnus Mid": list(adductor_magnus_mid_l.values()),
    "Adductor Magnus Proximal": list(adductor_magnus_proximal_l.values()),
    "Biceps Femoris Long Head": list(biceps_femoris_long_head_l.values()),
    "Gracilis": list(gracilis_l.values()),
    "Semitendinosus": list(semitendinosus_l.values()),
    "Semimembranosus": list(semimembranosus_l.values()),
}

hip_abduction_l_muscles = {
    "Gluteus Maximus": list(gluteus_maximus_l.values()),
    "Gluteus Medius": list(gluteus_medius_l.values()),
    "Gluteus Minimus": list(gluteus_minimus_l.values()),
    "Tensor Fasciae Latae": list(tensor_fasciae_latae_l.values()),
    "Piriformis": list(piriformis_l.values()),
    "Sartorius": list(sartorius_l.values()),
}

hip_flexion_l_muscles = {
    "Adductor Brevis": list(adductor_brevis_l.values()),
    "Adductor Longus": list(adductor_longus_l.values()),
    "Gluteus Minimus": list(gluteus_minimus_l.values()),
    "Gracilis": list(gracilis_l.values()),
    "Iliacus": list(iliacus_l.values()),
    "Psoas": list(psoas_l.values()),
    "Rectus Femoris": list(recfem_l.values()),
    "Sartorius": list(sartorius_l.values()),
    "Tensor Fasciae Latae": list(tensor_fasciae_latae_l.values()),
}

hip_extension_l_muscles = {
    "Adductor Longus": list(adductor_longus_l.values()),
    "Adductor Magnus Distal": list(adductor_magnus_distal_l.values()),
    "Adductor Magnus Ischial": list(adductor_magnus_ischial_l.values()),
    "Adductor Magnus Mid": list(adductor_magnus_mid_l.values()),
    "Adductor Magnus Proximal": list(adductor_magnus_proximal_l.values()),
    "Biceps Femoris Long Head": list(biceps_femoris_long_head_l.values()),
    "Gluteus Maximus": list(gluteus_maximus_l.values()),
    "Gluteus Medius": list(gluteus_medius_l.values()),
    "Gluteus Minimus": list(gluteus_minimus_l.values()),
    "Semitendinosus": list(semitendinosus_l.values()),
    "Semimembranosus": list(semimembranosus_l.values()),
}

hip_inrotation_l_muscles = {
    "Gluteus Medius": list(gluteus_medius_l.values()),
    "Gluteus Minimus": list(gluteus_minimus_l.values()),
    "Iliacus": list(iliacus_l.values()),
    "Psoas": list(psoas_l.values()),
    "Tensor Fasciae Latae": list(tensor_fasciae_latae_l.values()),
}

hip_exrotation_l_muscles = {
    "Gluteus Minimus": list(gluteus_minimus_l.values()),
    "Piriformis": list(piriformis_l.values()),
}

knee_flexion_l_muscles = {
    "Biceps Femoris Long Head": list(biceps_femoris_long_head_l.values()),
    "Biceps Femoris Short Head": list(biceps_fermoris_short_head_l.values()),
    "Gastrocnemius Lateral": list(gastrocnemius_lateral_l.values()),
    "Gastrocnemius Medial": list(gastrocnemius_medial_l.values()),
    "Gracilis": list(gracilis_l.values()),
    "Sartorius": list(sartorius_l.values()),
    "Semitendinosus": list(semitendinosus_l.values()),
    "Semimembranosus": list(semimembranosus_l.values()),
}

knee_extension_l_muscles = {
    "Rectus Femoris": list(recfem_l.values()),
    "Vastus Intermedius": list(vastus_intermedius_l.values()),
    "Vastus Lateralis": list(vastus_lateralis_l.values()),
    "Vastus Medialis": list(vastus_medialis_l.values()),
}

ankle_dorsiflexion_l_muscles = {
    "Extensor Digitorum Longus": list(extensor_digitorum_longus_l.values()),
    "Extensor Hallucis Longus": list(extensor_hallucis_longus_l.values()),
    "Tibialis Anterior": list(tibialis_anterior_l.values()),
}

ankle_plantarflexion_l_muscles = {
    "Flexor Digitorum Longus": list(flexor_digitorum_longus_l.values()),
    "Flexor Hallucis Longus": list(flexor_hallucis_longus_l.values()),
    "Gastrocnemius Lateral": list(gastrocnemius_lateral_l.values()),
    "Gastrocnemius Medial": list(gastrocnemius_medial_l.values()),
    "Peroneus Brevis": list(per_brev_l.values()),
    "Peroneus Longus": list(per_long_l.values()),
    "Soleus": list(soleus_l.values()),
    "Tibialis Posterior": list(tibialis_anterior_l.values()),
}

evertor_l_muscles = {
    "Extensor Digitorum Longus": list(extensor_digitorum_longus_l.values()),
    "Peroneus Brevis": list(per_brev_l.values()),
    "Peroneus Longus": list(per_long_l.values()),
}

invertor_l_muscles = {
    "Extensor Hallucis Longus": list(extensor_hallucis_longus_l.values()),
    "Flexor Digitorum Longus": list(flexor_digitorum_longus_l.values()),
    "Flexor Hallucis Longus": list(flexor_hallucis_longus_l.values()),
    "Tibialis Anterior": list(tibialis_anterior_l.values()),
    "Tibialis Posterior": list(tibialis_anterior_l.values()),
}

lu_muscle_groups_level0 = {
    "left Extensor Hallucis Longus": list(extensor_hallucis_longus_l.values()),
    "left Flexor Digitorum Longus": list(flexor_digitorum_longus_l.values()),
    "left Flexor Hallucis Longus": list(flexor_hallucis_longus_l.values()),
    "left Tibialis Anterior": list(tibialis_anterior_l.values()),
    "left Tibialis Posterior": list(tibialis_anterior_l.values()),
    "left Extensor Digitorum Longus": list(extensor_digitorum_longus_l.values()),
    "left Peroneus": list(per_brev_l.values()) + list(per_long_l.values()),
    "left Gastrocnemius": list(gastrocnemius_lateral_l.values()) + list(gastrocnemius_medial_l.values()),
    "left Soleus": list(soleus_l.values()),
    "left Rectus Femoris": list(recfem_l.values()),
    "left Vastus Intermedius": list(vastus_intermedius_l.values()),
    "left Vastus Lateralis": list(vastus_lateralis_l.values()),
    "left Vastus Medialis": list(vastus_medialis_l.values()),
    "left Biceps Femoris Long Head": list(biceps_femoris_long_head_l.values()),
    "left Biceps Femoris Short Head": list(biceps_fermoris_short_head_l.values()),
    "left Gracilis": list(gracilis_l.values()),
    "left Sartorius": list(sartorius_l.values()),
    "left Semitendinosus": list(semitendinosus_l.values()),
    "left Semimembranosus": list(semimembranosus_l.values()),
    "left Gluteus Minimus": list(gluteus_minimus_l.values()),
    "left Piriformis": list(piriformis_l.values()),
    "left Gluteus Medius": list(gluteus_medius_l.values()),
    "left Iliacus": list(iliacus_l.values()),
    "left Psoas": list(psoas_l.values()),
    "left Tensor Fasciae Latae": list(tensor_fasciae_latae_l.values()),
    "left Adductor Brevis": list(adductor_brevis_l.values()),
    "left Adductor Longus": list(adductor_longus_l.values()),
    "left Adductor Magnus": list(adductor_magnus_distal_l.values())+list(adductor_magnus_ischial_l.values())+list(adductor_magnus_mid_l.values())+list(adductor_magnus_proximal_l.values()),
    "left Gluteus Maximus": list(gluteus_maximus_l.values()),

    "right Extensor Hallucis Longus": list(extensor_hallucis_longus_r.values()),
    "right Flexor Digitorum Longus": list(flexor_digitorum_longus_r.values()),
    "right Flexor Hallucis Longus": list(flexor_hallucis_longus_r.values()),
    "right Tibialis Anterior": list(tibialis_anterior_r.values()),
    "right Tibialis Posterior": list(tibialis_anterior_r.values()),
    "right Extensor Digitorum Longus": list(extensor_digitorum_longus_r.values()),
    "right Peroneus": list(per_brev_r.values()) + list(per_long_r.values()),
    "right Gastrocnemius": list(gastrocnemius_lateral_r.values()) + list(gastrocnemius_medial_r.values()),
    "right Soleus": list(soleus_r.values()),
    "right Rectus Femoris": list(recfem_r.values()),
    "right Vastus Intermedius": list(vastus_intermedius_r.values()),
    "right Vastus Lateralis": list(vastus_lateralis_r.values()),
    "right Vastus Medialis": list(vastus_medialis_r.values()),
    "right Biceps Femoris Long Head": list(biceps_femoris_long_head_r.values()),
    "right Biceps Femoris Short Head": list(biceps_fermoris_short_head_r.values()),
    "right Gracilis": list(gracilis_r.values()),
    "right Sartorius": list(sartorius_r.values()),
    "right Semitendinosus": list(semitendinosus_r.values()),
    "right Semimembranosus": list(semimembranosus_r.values()),
    "right Gluteus Minimus": list(gluteus_minimus_r.values()),
    "right Piriformis": list(piriformis_r.values()),
    "right Gluteus Medius": list(gluteus_medius_r.values()),
    "right Iliacus": list(iliacus_r.values()),
    "right Psoas": list(psoas_r.values()),
    "right Tensor Fasciae Latae": list(tensor_fasciae_latae_r.values()),
    "right Adductor Brevis": list(adductor_brevis_r.values()),
    "right Adductor Longus": list(adductor_longus_r.values()),
    "right Adductor Magnus": list(adductor_magnus_distal_r.values()) + list(adductor_magnus_ischial_r.values()) + list(adductor_magnus_mid_r.values()) + list(adductor_magnus_proximal_r.values()),
    "right Gluteus Maximus": list(gluteus_maximus_r.values()),
}

lu_muscle_groups_level1 = {
    "right invertor": list(chain.from_iterable(invertor_r_muscles.values())),
    "right evertor": list(chain.from_iterable(evertor_r_muscles.values())),
    "right ankle plantarflexion": list(
        chain.from_iterable(ankle_plantarflexion_r_muscles.values())
    ),
    "right ankle dorsiflexion": list(
        chain.from_iterable(ankle_dorsiflexion_r_muscles.values())
    ),
    "right knee extension": list(
        chain.from_iterable(knee_extension_r_muscles.values())
    ),
    "right knee flexion": list(chain.from_iterable(knee_flexion_r_muscles.values())),
    "right hip external rotation": list(
        chain.from_iterable(hip_exrotation_r_muscles.values())
    ),
    "right hip internal rotation": list(
        chain.from_iterable(hip_inrotation_r_muscles.values())
    ),
    "right hip extension": list(chain.from_iterable(hip_extension_r_muscles.values())),
    "right hip flexion": list(chain.from_iterable(hip_flexion_r_muscles.values())),
    "right hip abduction": list(chain.from_iterable(hip_abduction_r_muscles.values())),
    "right hip adduction": list(chain.from_iterable(hip_adduction_r_muscles.values())),
    
    "left invertor": list(chain.from_iterable(invertor_l_muscles.values())),
    "left evertor": list(chain.from_iterable(evertor_l_muscles.values())),
    "left ankle plantarflexion": list(
        chain.from_iterable(ankle_plantarflexion_l_muscles.values())
    ),
    "left ankle dorsiflexion": list(
        chain.from_iterable(ankle_dorsiflexion_l_muscles.values())
    ),
    "left knee extension": list(chain.from_iterable(knee_extension_l_muscles.values())),
    "left knee flexion": list(chain.from_iterable(knee_flexion_l_muscles.values())),
    "left hip external rotation": list(
        chain.from_iterable(hip_exrotation_l_muscles.values())
    ),
    "left hip internal rotation": list(
        chain.from_iterable(hip_inrotation_l_muscles.values())
    ),
    "left hip extension": list(chain.from_iterable(hip_extension_l_muscles.values())),
    "left hip flexion": list(chain.from_iterable(hip_flexion_l_muscles.values())),
    "left hip abduction": list(chain.from_iterable(hip_abduction_l_muscles.values())),
    "left hip adduction": list(chain.from_iterable(hip_adduction_l_muscles.values())),
}

lu_muscle_groups_level2 = {
    "right ankle": list(
        set(
            lu_muscle_groups_level1["right invertor"]
            + lu_muscle_groups_level1["right evertor"]
            + lu_muscle_groups_level1["right ankle plantarflexion"]
            + lu_muscle_groups_level1["right ankle dorsiflexion"]
        )
    ),
    "right knee": list(
        set(
            lu_muscle_groups_level1["right knee extension"]
            + lu_muscle_groups_level1["right knee flexion"]
        )
    ),
    "right hip": list(
        set(
            lu_muscle_groups_level1["right hip external rotation"]
            + lu_muscle_groups_level1["right hip internal rotation"]
            + lu_muscle_groups_level1["right hip extension"]
            + lu_muscle_groups_level1["right hip flexion"]
            + lu_muscle_groups_level1["right hip abduction"]
            + lu_muscle_groups_level1["right hip adduction"]
        )
    ),
    "left ankle": list(
        set(
            lu_muscle_groups_level1["left invertor"]
            + lu_muscle_groups_level1["left evertor"]
            + lu_muscle_groups_level1["left ankle plantarflexion"]
            + lu_muscle_groups_level1["left ankle dorsiflexion"]
        )
    ),
    "left knee": list(
        set(
            lu_muscle_groups_level1["left knee extension"]
            + lu_muscle_groups_level1["left knee flexion"]
        )
    ),
    "left hip": list(
        set(
            lu_muscle_groups_level1["left hip external rotation"]
            + lu_muscle_groups_level1["left hip internal rotation"]
            + lu_muscle_groups_level1["left hip extension"]
            + lu_muscle_groups_level1["left hip flexion"]
            + lu_muscle_groups_level1["left hip abduction"]
            + lu_muscle_groups_level1["left hip adduction"]
        )
    ),
}

lu_muscle_groups_level3 = {
    "right lower limb": list(
        set(
            lu_muscle_groups_level2["right ankle"]
            + lu_muscle_groups_level2["right knee"]
            + lu_muscle_groups_level2["right hip"]
        )
    ),
    "left lower limb": list(
        set(
            lu_muscle_groups_level2["left ankle"]
            + lu_muscle_groups_level2["left knee"]
            + lu_muscle_groups_level2["left hip"]
        )
    ),
}

lu_muscle_groups_level4 = {
    "lower limbs": list(
        set(
            lu_muscle_groups_level3["right lower limb"]
            + lu_muscle_groups_level3["left lower limb"]
        )
    ),
}


class LowerBodyMuscleGroups:
    def __init__(self):
        self.level0 = lu_muscle_groups_level0
        self.level1 = lu_muscle_groups_level1
        self.level2 = lu_muscle_groups_level2
        self.level3 = lu_muscle_groups_level3
        self.level4 = lu_muscle_groups_level4

    def get_muscle_group(self, level, group):
        if level == 0:
            return self.level0[group]
        elif level == 1:
            return self.level1[group]
        elif level == 2:
            return self.level2[group]
        elif level == 3:
            return self.level3[group]
        elif level == 4:
            return self.level4[group]
        else:
            return None

    def get_muscle_groups(self, level):
        if level == 0:
            return self.level0
        elif level == 1:
            return self.level1
        elif level == 2:
            return self.level2
        elif level == 3:
            return self.level3
        elif level == 4:
            return self.level4
        else:
            return None


if __name__ == "__main__":
    lbmg = LowerBodyMuscleGroups()
    print(lbmg.get_muscle_group(1, "right invertor"))
    print(lbmg.get_muscle_groups(1))
    print(lbmg.get_muscle_group(2, "right ankle"))
    print(lbmg.get_muscle_groups(2))
    print(lbmg.get_muscle_group(3, "right lower limb"))
    print(lbmg.get_muscle_groups(3))
    print(lbmg.get_muscle_group(4, "lower limbs"))
    print(lbmg.get_muscle_groups(4))
