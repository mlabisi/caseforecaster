import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

public class CaseForecaster {
    private static LinearRegressionModel countyModel;
    private static LinearRegressionModel stateModel;
    private static PipelineModel cModel;
    private static PipelineModel sModel;

    private static SparkSession spark = SparkSession.builder().master("local[*]").appName("Case Predictor").getOrCreate();
    private static Map<String, Integer> locationToFIPS = new HashMap<>();
    private static Map<String, Integer> dateToIndex = new HashMap<>();

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
    private static final String COUNTIES_FILE = "covid-19-counties.csv";
    private static final String STATES_FILE = "covid-19-states.csv";

    // configure csv input schema
    private static StructType stateSchema = new StructType(
            new StructField[]{
                    new StructField("date", DataTypes.StringType, false, Metadata.empty()),
                    new StructField("state", DataTypes.StringType, false, Metadata.empty()),
                    new StructField("fips", DataTypes.IntegerType, false, Metadata.empty()),
                    new StructField("cases", DataTypes.IntegerType, false, Metadata.empty()),
                    new StructField("deaths", DataTypes.IntegerType, false, Metadata.empty())
            }
    );

    private static StructType countySchema = new StructType(
            new StructField[]{
                    new StructField("date", DataTypes.StringType, false, Metadata.empty()),
                    new StructField("county", DataTypes.StringType, false, Metadata.empty()),
                    new StructField("state", DataTypes.StringType, false, Metadata.empty()),
                    new StructField("fips", DataTypes.IntegerType, false, Metadata.empty()),
                    new StructField("cases", DataTypes.IntegerType, false, Metadata.empty()),
                    new StructField("deaths", DataTypes.IntegerType, false, Metadata.empty())
            }
    );

    private static void gatherData(String filename, String url) {
        System.out.println("Generating data from given source: " + url);
        boolean inCountyMode = filename.contains("counties");
        try {
            File dir = new File(DATA_PATH);
            dir.mkdir();

            // store fresh copy of dataset in our cache
            File file = new File(DATA_PATH + filename);
            FileUtils.copyURLToFile(new URL(url), file);
            System.out.println("Successfully created " + DATA_PATH + filename);

            // load the data as a Dataframe
            Dataset<Row> data = spark.read().format("csv").option("header", true).schema(inCountyMode ? countySchema : stateSchema).load(DATA_PATH + filename);

            // build up location to FIPS index and "double-ize" dates
            data.foreach(row -> {
                if ((inCountyMode ? row.get(3) : row.get(2)) != null) {
                    String location = (inCountyMode ? row.getString(1) + ", " + row.getString(2) : row.getString(1));
                    int fips = (inCountyMode ? row.getInt(3) : row.getInt(2));
                    locationToFIPS.put(location, fips);
                }
            });

            // remove location since we have the fips
            data.drop("state");
            if (inCountyMode) {
                data.drop("county");
            }

            // split all data into training data and testing data
            Dataset<Row>[] splits = data.randomSplit(new double[]{0.65, 0.35});
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testingData = splits[1];

            // configure pipeline
            StringIndexer indexer = new StringIndexer()
                    .setInputCol("date")
                    .setOutputCol("indexedDate")
                    .setHandleInvalid("skip");
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(new String[]{"indexedDate", "fips"})
                    .setOutputCol("features")
                    .setHandleInvalid("skip");
            DecisionTreeRegressor linReg = new DecisionTreeRegressor()
                    .setFeaturesCol("features")
                    .setLabelCol("cases")
                    .setPredictionCol("prediction")
                    .setVarianceCol("var")
                    .setMaxBins(Integer.MAX_VALUE);
            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{indexer, assembler, linReg});

            // train model
            // create linear regression model
            if (inCountyMode) {
                cModel = pipeline.fit(trainingData);
            } else {
                sModel = pipeline.fit(trainingData);
            }

            // test model
            Dataset<Row> predictions = inCountyMode ? cModel.transform(testingData): sModel.transform(testingData);
            predictions.show(false);

            // Select (prediction, true label) and compute test error.
            RegressionEvaluator evaluator = new RegressionEvaluator()
                    .setLabelCol("cases")
                    .setPredictionCol("prediction")
                    .setMetricName("rmse");
            double rmse = evaluator.evaluate(predictions);
            System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
        } catch (MalformedURLException e) {
            System.out.println("Erroneous URL provided.");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("File could not be created.");
            e.printStackTrace();
        }
    }

    // take in a county and a date --> predict cases
    // perhaps give a dropdown to select county,
    // and calendar excluding today and past
    private static void makePrediction(String location, String date) {
        boolean inCountyMode = location.contains(",");
        int fips = getFIPS(location);
        List<Row> inputData;

        // prepare user input for pipeline
        if (inCountyMode) {
            String[] splitLocation = location.replace(",", "").split("\\s+");
            String county = splitLocation[0];
            String state = splitLocation[1];

            inputData = Collections.singletonList(RowFactory.create(date, county, state, fips, 0, 0));
        } else {
            inputData = Collections.singletonList(RowFactory.create(date, location, fips, 0, 0));
        }

        Dataset<Row> input = spark.createDataFrame(inputData, inCountyMode ? countySchema : stateSchema);

        Dataset<Row> output = inCountyMode ? cModel.transform(input): sModel.transform(input);
        output.select("cases", "prediction").show(false);
//        (inCountyMode ? cModel.transform(input) : sModel.transform(input)).show(false);

//        int casesCt = (inCountyMode ? cModel.transform(input) : sModel.transform(input)).select("prediction").collectAsList().get(0).getInt(0);
//        System.out.println("I predict there will be " + casesCt + " cases in " + location + " on " + date + ".");
    }

    private static int getFIPS(String location) {
        return locationToFIPS.get(location);
    }

    private static void runModels() {
        // gather data
        // prepare data
        // collect training data
        // configure pipeline tokenizer
        // configure pipeline transformer
        // configure pipeline model
        // fit training data
        // collect testing data
        // predict using testing data
        gatherData(COUNTIES_FILE, COUNTRIES_URL);
        gatherData(STATES_FILE, STATES_URL);
    }

    public static void main(String[] args) {
        runModels();

//        makePrediction("Snohomish, Washington", "2020-04-03");
//        makePrediction("Los Angeles, California", "2020-04-03");
//        makePrediction("Hawaii", "2020-04-03");
//        makePrediction("New York", "2020-04-03");

        while (true) {
            System.out.print("");
        }

        // TODO: GUI for makePrediction
    }

    private static void chooseModel() {

    }
}