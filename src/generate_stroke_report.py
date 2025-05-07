import os
import logging
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, mean, stddev
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_stroke_report(data_path="data/healthcare-dataset-stroke-data.csv"):
    """Generate all outputs for the stroke prediction project report."""
    spark = None
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("StrokeReportGeneration") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
        logger.info("Spark session initialized")

        # Create output directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("images", exist_ok=True)
        logger.info("Output directories data/ and images/ created or verified")

        # Load dataset
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        df = df.repartition(200)
        logger.info(f"Dataset loaded with {df.rdd.getNumPartitions()} partitions")

        # Cast bmi to FloatType, handling non-numeric values
        df = df.withColumn("bmi", when(col("bmi").cast(FloatType()).isNotNull(), col("bmi").cast(FloatType())).otherwise(None))
        logger.info("bmi column cast to FloatType")

        # Impute missing BMI values
        imputer = Imputer(inputCols=["bmi"], outputCols=["bmi_imputed"]).setStrategy("mean")
        df = imputer.fit(df).transform(df)
        logger.info("Imputed missing BMI values")

        # Cache DataFrame for performance
        df.cache()
        logger.info("DataFrame cached for performance")

        # Summary statistics
        numeric_cols = ["age", "avg_glucose_level", "bmi_imputed"]
        stats = df.select([mean(col(c)).alias(f"mean_{c}") for c in numeric_cols] +
                         [stddev(col(c)).alias(f"stddev_{c}") for c in numeric_cols]).collect()[0]
        stats_df = pd.DataFrame({k: [stats[k]] for k in stats.asDict()})
        stats_df.to_csv("data/stat_tests.csv", index=False)
        logger.info("Summary statistics saved to data/stat_tests.csv")

        # Correlation analysis
        corr_data = []
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 <= col2:
                    corr = df.stat.corr(col1, col2)
                    corr_data.append({"var1": col1, "var2": col2, "correlation": corr})
        corr_df = pd.DataFrame(corr_data)
        corr_df.to_csv("data/correlations.csv", index=False)
        logger.info("Correlation matrix saved to data/correlations.csv")

        # Stroke rates by categorical variables
        cat_cols = ["hypertension", "smoking_status"]
        for cat in cat_cols:
            stroke_rates = df.groupBy(cat, "stroke").agg(count("*").alias("count")) \
                            .groupBy(cat).pivot("stroke").sum("count").fillna(0) \
                            .withColumn("stroke_rate", col("1") / (col("0") + col("1")))
            stroke_rates_pd = stroke_rates.toPandas()
            fig = px.bar(stroke_rates_pd, x=cat, y="stroke_rate", title=f"Stroke Rate by {cat.replace('_', ' ').title()}")
            fig.update_layout(
                yaxis_title="Stroke Rate",
                xaxis_title=cat.replace("_", " ").title(),
                title_x=0.5,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="black", family="Arial", size=12),
                title_font_size=14,
                hovermode="closest",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"),
                bargap=0.2,
                bargroupgap=0.1,
                yaxis_range=[0, stroke_rates_pd["stroke_rate"].max() * 1.1],
                margin=dict(l=50, r=50, t=50, b=50),
                width=600,
                height=400,
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12), gridcolor="lightgray", zerolinecolor="black", zerolinewidth=1, showline=True, linewidth=1, linecolor="black"),
                xaxis_showline=True,
                xaxis_linewidth=1,
                xaxis_linecolor="black",
                yaxis_showline=True,
                yaxis_linewidth=1,
                yaxis_linecolor="black"
            )
            fig.update_traces(marker_color="#1f77b4", text=stroke_rates_pd["stroke_rate"].round(3), textposition="auto")
            fig.write_image(f"images/{cat}_vs_stroke.png")
            logger.info(f"Visualization saved to images/{cat}_vs_stroke.png")

        # Cast numerical columns for training
        numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
        for col_name in numeric_cols:
            df = df.withColumn(col_name, when(col(col_name).cast(FloatType()).isNotNull(), col(col_name).cast(FloatType())).otherwise(None))
        logger.info("Numerical columns cast to FloatType for training")

        # Handle class imbalance (oversampling minority class)
        positive_count = df.filter(col("stroke") == 1).count()
        negative_count = df.filter(col("stroke") == 0).count()
        balance_ratio = negative_count / positive_count
        df_positive = df.filter(col("stroke") == 1).sample(True, balance_ratio, seed=42)
        df = df.filter(col("stroke") == 0).union(df_positive)
        logger.info(f"Class imbalance handled with oversampling (ratio: {balance_ratio:.2f})")

        # Split data
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        logger.info("Dataset split into 80% training and 20% testing")

        # Define pipeline
        categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in categorical_cols]
        assembler = VectorAssembler(
            inputCols=[f"{col}_index" for col in categorical_cols] + numeric_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        classifier = GBTClassifier(labelCol="stroke", featuresCol="features", maxIter=20, seed=42)
        pipeline = Pipeline(stages=indexers + [assembler, classifier])

        # Train model
        model = pipeline.fit(train_df)
        logger.info("Trained GBTClassifier model")

        # Make predictions
        predictions = model.transform(test_df)
        predictions = predictions.withColumn("stroke_risk", col("probability").getItem(1))
        logger.info("Generated predictions on test set")

        # Save predictions
        predictions.select("age", "avg_glucose_level", "bmi", "stroke", "prediction", "stroke_risk") \
            .toPandas().to_csv("data/predictions.csv", index=False)
        logger.info("Saved predictions to data/predictions.csv")

        # Evaluate model
        evaluator = BinaryClassificationEvaluator(labelCol="stroke", rawPredictionCol="prediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        logger.info(f"Model AUC: {auc:.4f}")

        # Save evaluation metrics
        eval_df = pd.DataFrame([{"model": "GBTClassifier", "AUC": auc}])
        eval_df.to_csv("data/evaluation_metrics.csv", index=False)
        logger.info("Evaluation metrics saved to data/evaluation_metrics.csv")

        # Save model comparison (single model for simplicity)
        eval_df.to_csv("data/model_comparisons.csv", index=False)
        logger.info("Model comparison metrics saved to data/model_comparisons.csv")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise
    finally:
        if spark is not None:
            spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    generate_stroke_report()
