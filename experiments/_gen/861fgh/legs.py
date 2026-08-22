"""Leg-specific metadata for the V3-EXQ-861f/861g/861h GOV-FANOUT-1 portfolio."""

QID = "inv050_mech180_861e_producer_vs_intervention_isolation"
AUTOPSY = "failure_autopsy_V3-EXQ-861e_2026-08-21"
REF_17 = "17befb8c46f0b7352f74a6b6e3ee4fc9715878fc"
REF_F8 = "f810969089fa8193959f49072e8aa1c2de0cb193"
RUN_861E = ("v3_exq_861e_inv050_mech180_calibration_power_raised_replication"
            "_20260820T214522Z_v3")
RUN_861C = ("v3_exq_861c_inv050_mech180_calibration_fixed_replication"
            "_20260814T231404Z_v3")

LEGS = {
    "861f": dict(
        qid_hyp="H1",
        axis="measurement",
        letter="f",
        exp_type=("v3_exq_861f_inv050_mech180_h1_measurement_rng_isolation"),
        queue_id="V3-EXQ-861f",
        pin_ref=REF_17,
        pin_ref_short="17befb8c",
        marker_present=True,
        title=("H1 leg (GOV-FANOUT-1, measurement axis): is the V3-EXQ-861e seed-271 "
               "HIGH-arm MEL collapse an intervention-isolation defect? Reseed the "
               "measurement phase, substrate pinned to 861e's own 17befb8c"),
        variant_a="reseeded",
        variant_b="unreseeded",
    ),
    "861g": dict(
        qid_hyp="H3",
        axis="algorithm",
        letter="g",
        exp_type=("v3_exq_861g_inv050_mech180_h3_substrate_pin_f810969"),
        queue_id="V3-EXQ-861g",
        pin_ref=REF_F8,
        pin_ref_short="f810969",
        marker_present=False,
        title=("H3 leg (GOV-FANOUT-1, algorithm axis): does the V3-EXQ-861e seed-271 "
               "HIGH-arm MEL collapse survive on 861c's substrate? 861e protocol "
               "(CALIB_DRAWS=10 + R3) pinned to f810969"),
        variant_a="n10",
        variant_b="n5",
    ),
    "861h": dict(
        qid_hyp="CONTROL",
        axis="representation",
        letter="h",
        exp_type=("v3_exq_861h_inv050_mech180_contextmemory_write_lock_control"),
        queue_id="V3-EXQ-861h",
        pin_ref=REF_17,
        pin_ref_short="17befb8c",
        marker_present=True,
        title=("Substrate-defect CONTROL leg (GOV-FANOUT-1 Step 2.5b(iv) coverage "
               "gap; 2.5c corrupting overlap): is the decisive seed's MEL readout "
               "trustworthy at all? Repeat the 861e protocol on 17befb8c with the "
               "already-built non-degenerate ContextMemory write selection ON"),
        variant_a="refractory",
        variant_b="argmin_legacy",
    ),
}
