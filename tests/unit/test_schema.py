from matscreen.data.schema import (
    DataSource,
    MaterialCard,
    MaterialRecord,
    PredictionWithCI,
    PropertySet,
    SymmetryInfo,
)


def test_material_record_roundtrip():
    record = MaterialRecord(
        material_id="mp-149",
        source=DataSource.MATERIALS_PROJECT,
        formula="Si",
        symmetry=SymmetryInfo(crystal_system="cubic", space_group="Fd-3m"),
        properties=PropertySet(band_gap=1.11, formation_energy_per_atom=0.0),
    )
    data = record.model_dump()
    restored = MaterialRecord.model_validate(data)
    assert restored.material_id == "mp-149"
    assert restored.properties.band_gap == 1.11
    assert restored.symmetry.crystal_system == "cubic"


def test_property_set_optional_fields():
    props = PropertySet(band_gap=1.5)
    assert props.band_gap == 1.5
    assert props.formation_energy_per_atom is None
    assert props.bulk_modulus_kv is None


def test_prediction_with_ci():
    pred = PredictionWithCI(
        mean=1.42,
        std=0.08,
        ci_lower=1.26,
        ci_upper=1.58,
        unit="eV",
        calibrated=True,
    )
    assert pred.calibrated is True
    assert pred.mean == 1.42


def test_material_card():
    card = MaterialCard(
        material_id="mp-2534",
        formula="GaAs",
        source=DataSource.MATERIALS_PROJECT,
        symmetry=SymmetryInfo(crystal_system="cubic"),
        predictions={
            "bandgap": PredictionWithCI(
                mean=1.42, std=0.08, ci_lower=1.26, ci_upper=1.58, unit="eV"
            ),
        },
        pareto_rank=1,
        pareto_front=0,
    )
    assert card.pareto_rank == 1
    assert "bandgap" in card.predictions
