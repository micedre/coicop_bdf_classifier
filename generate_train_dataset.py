import duckdb

duckdb.sql(
    """
    CREATE VIEW ddc_sample AS
    FROM 's3://travail/projet-ml-classification-bdf/confidentiel/personnel_sensible/data/raw/sample/ddc_20251217.parquet'
    """
)

duckdb.sql(
    """
    CREATE VIEW ddc_sample_ecoicopv2 AS
    SELECT description_ean as product, code_sous_classe_ecoicopv2 as code
    FROM ddc_sample INNER JOIN 'data/table_passage_coicop.csv' as table_passage
    ON ddc_sample.variete = table_passage.code_variete_ecoicopv1
    """
)

duckdb.sql(
    """
    CREATE TABLE coicop_train_dataset AS
    FROM ddc_sample_ecoicopv2
    UNION
    FROM 'data/synthetic_data.csv' select product, code   
    """
)

duckdb.sql(
    """
    FROM coicop_train_dataset LIMIT 10;
    """
).show()

duckdb.sql(
    """
    FROM coicop_train_dataset
    """
).to_parquet('coicop_train_dataset.paquet')