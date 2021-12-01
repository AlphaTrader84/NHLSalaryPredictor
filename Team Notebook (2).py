# Databricks notebook source
# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

from pyspark.sql.types import ArrayType, DoubleType, IntegerType, LongType, StringType, StructType, StructField, TimestampType

userDefinedSchema = StructType([ \
                               StructField("Salary", DoubleType(), True),
                               StructField("Cntry", StringType(), True), \
                               StructField("Ht", DoubleType(), True), \
                               StructField("Wt", DoubleType(), True), \
                               StructField("DftRd", DoubleType(), True), \
                               StructField("Ovrl", DoubleType(), True), \
                               StructField("Position", StringType(), True), \
                               StructField("GP", DoubleType(), True), \
                               StructField("G", DoubleType(), True), \
                               StructField("A", DoubleType(), True), \
                               StructField("PTS", DoubleType(), True), \
                               StructField("+/-", DoubleType(), True), \
                               StructField("Shifts", DoubleType(), True), \
                               StructField("TOI", DoubleType(), True)
                               ])

# File location and type
file_location = "dbfs:/FileStore/shared_uploads/jmgreenlee@usf.edu/train__reduced_.csv"
file_type = "csv"

# CSV options
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.csv(file_location, header=True, nullValue="NA", schema=userDefinedSchema)

df.dropna()
df = df.drop(df.Cntry).drop(df.Position)

# Check for missing values 
from pyspark.sql.functions import isnan, isnull, when, count, col

df.select([count(when(isnan(c) | isnull(c), c)).alias(c) for c in df.columns]).show()



# COMMAND ----------

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "train__reduced__csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `train__reduced__csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "train__reduced__csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

train_data,test_data=df.randomSplit([0.7,0.3])

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler,StringIndexer,OneHotEncoder
from pyspark.ml import Pipeline

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(missingValue=0, inputCols=['DftRd', 'Ovrl'], outputCols=['imp_DftRd', 'imp_Ovrl'])

# COMMAND ----------

#Vector Assembler used to create vector of input features
assembler = VectorAssembler(inputCols=['Salary','Ht','Wt','imp_DftRd','imp_Ovrl','GP','G','A','PTS','+/-','Shifts','TOI'],
                           outputCol='features')

# COMMAND ----------

lm = LinearRegression(featuresCol='features', labelCol='Salary', predictionCol='prediction', regParam=0.0, solver="normal")

# COMMAND ----------

pipe = Pipeline(stages=[imputer, assembler, lm])
#pipe = Pipeline(stages=[country_indexer, country_encoder, position_indexer, position_encoder, assembler, lm])
#data_encoder,country_indexer,position_indexer


# COMMAND ----------
fit_model=pipe.fit(train_data).transform(train_data)

fit_model


# COMMAND ----------

fit_model.show()

# COMMAND ----------



# COMMAND ----------

fit_model.transform(train_data)


# COMMAND ----------

#spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
fit_model=pipe.fit(train_data)
