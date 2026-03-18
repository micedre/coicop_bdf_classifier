
CREATE SECRET secret_prod (
    TYPE S3,
    KEY_ID getenv('MINIO_PROD_ACCESS_KEY_ID'),
    SECRET getenv('MINIO_PROD_SECRET_ACCESS_KEY'),
    ENDPOINT getenv('MINIO_PROD_S3_ENDPOINT'),
    SESSION_TOKEN '',
    REGION 'us-east-1',
    URL_STYLE 'path',
    SCOPE 's3://projet-ddc/'
);

CREATE SECRET secret_ls3 (
    TYPE S3,
    KEY_ID getenv('AWS_ACCESS_KEY_ID'),
    SECRET getenv('AWS_SECRET_ACCESS_KEY'),
    ENDPOINT getenv('AWS_S3_ENDPOINT'),
    SESSION_TOKEN getenv('AWS_SESSION_TOKEN'),
    REGION 'us-east-1',
    URL_STYLE 'path',
    SCOPE 's3://travail/projet-ml-classification-bdf'
);

SET memory_limit = '6GB';


CREATE VIEW famille_circana AS FROM 'data/famille_circana.csv';

CREATE VIEW sample_122025 AS 
    SELECT ddc.description_ean, 
           ddc.variete, 
           CASE WHEN ddc.variete[:2]='99' 
                THEN famille_circana.coicop 
                ELSE ddc.variete 
            END AS coicop_code
    FROM 
        (SELECT DISTINCT description_ean, 
                variete, 
                id_famille 
        FROM 
        's3://projet-ddc/protected/iceberg-warehouse-prod/entrepot_ddc/complement_collecte/data/annee=2025/mois=12/**/*.parquet') ddc  
        LEFT JOIN famille_circana 
        ON ddc.id_famille=famille_circana.id_famille
    WHERE len(coicop_code)>=10 and coicop_code[:2] != '99';

CREATE VIEW sample_012026 AS 
    SELECT ddc.description_ean, 
           ddc.variete, 
           CASE WHEN ddc.variete[:2]='99' 
                THEN famille_circana.coicop 
                ELSE ddc.variete 
            END AS coicop_code 
    FROM 
        (SELECT DISTINCT description_ean, 
                variete, 
                id_famille 
        FROM 
        's3://projet-ddc/protected/iceberg-warehouse-prod/entrepot_ddc/complement_collecte/data/annee=2026/**/*.parquet') ddc  
        LEFT JOIN famille_circana 
        ON ddc.id_famille=famille_circana.id_famille
    WHERE len(coicop_code)>=10 and coicop_code[:2] != '99';

CREATE VIEW sample AS 
    FROM sample_122025 
    UNION BY NAME
    FROM sample_012026;


COPY 'sample' to 's3://travail/projet-ml-classification-bdf/confidentiel/personnel_sensible/data/raw/sample/ddc_sample.parquet';